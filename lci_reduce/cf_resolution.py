"""Characterisation factor ambiguity resolution helpers."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Tuple

from .models import CharacterisationFactorCandidate


def _normalise_text(value: Optional[str]) -> str:
    if value is None:
        return ""
    return " ".join(value.strip().lower().split())


def _normalise_float(value: float) -> str:
    return format(float(value), ".15g")


def _json_array(values: Sequence[object]) -> str:
    return json.dumps(list(values), ensure_ascii=True, sort_keys=True)


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
class CFPromptResult:
    action: str
    candidate_index: Optional[int] = None


@dataclass
class CFResolutionChoiceRecord:
    category_id: str
    category_name: str
    method_id: str
    method_name: str
    flow_id: str
    flow_name: str
    process_id: str
    process_name: str
    exchange_id: str
    exchange_index: str
    chosen_cf_value: str
    chosen_candidate_metadata: str
    rejected_cf_values: str
    rejected_candidate_metadata: str
    all_candidate_cf_values: str
    all_candidate_metadata: str
    choice_origin: str
    timestamp: str


@dataclass
class CFResolutionSummary:
    n_cf_ambiguities_found: int = 0
    n_cf_ambiguity_keys_unique: int = 0
    n_cf_ambiguities_resolved_automatically: int = 0
    n_cf_unique_user_decisions: int = 0
    n_cf_ambiguities_resolved_by_user_choice: int = 0
    n_cf_resolution_choices_reused: int = 0
    n_cf_ambiguities_unresolved: int = 0


@dataclass
class CFResolutionDecision:
    status: str
    candidate: Optional[CharacterisationFactorCandidate]
    reason: str
    audit_record: Optional[CFResolutionChoiceRecord] = None


PromptCallback = Callable[[CFAmbiguityContext, Sequence[CharacterisationFactorCandidate]], CFPromptResult]


CHOICE_FILE_COLUMNS = [
    "category_id",
    "category_name",
    "method_id",
    "method_name",
    "flow_id",
    "flow_name",
    "process_id",
    "process_name",
    "exchange_id",
    "exchange_index",
    "chosen_cf_value",
    "chosen_candidate_metadata",
    "rejected_cf_values",
    "rejected_candidate_metadata",
    "all_candidate_cf_values",
    "all_candidate_metadata",
    "choice_origin",
    "timestamp",
]


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


def candidate_metadata_json(candidate: CharacterisationFactorCandidate) -> str:
    return json.dumps(candidate_metadata(candidate), ensure_ascii=True, sort_keys=True)


def candidate_display_text(candidate: CharacterisationFactorCandidate) -> str:
    metadata = candidate_metadata(candidate)
    lines = [
        f"CF value: {_normalise_float(candidate.cf_value)}",
        f"Unit: {metadata['cf_unit'] or '-'}",
        f"Flow property: {metadata['cf_flow_property_name'] or metadata['cf_flow_property_id'] or '-'}",
        f"Compartment: {metadata['cf_compartment'] or '-'}",
        f"Subcompartment: {metadata['cf_subcompartment'] or '-'}",
        f"Location: {metadata['cf_location_name'] or metadata['cf_location_id'] or '-'}",
        f"Region: {metadata['cf_region'] or '-'}",
        f"Source path: {metadata['source_file'] or '-'}",
    ]
    return "\n".join(lines)


def candidate_values_json(candidates: Sequence[CharacterisationFactorCandidate]) -> str:
    return _json_array([_normalise_float(candidate.cf_value) for candidate in candidates])


def candidate_metadata_list_json(candidates: Sequence[CharacterisationFactorCandidate]) -> str:
    return _json_array([candidate_metadata(candidate) for candidate in candidates])


def _normalised_candidate_identity(candidate: CharacterisationFactorCandidate) -> Tuple[str, str]:
    metadata = {
        key: _normalise_text(str(value) if value is not None else "")
        for key, value in candidate_metadata(candidate).items()
        if key != "raw_factor_object"
    }
    return _normalise_float(candidate.cf_value), json.dumps(metadata, ensure_ascii=True, sort_keys=True)


def _normalised_choice_identity(record: CFResolutionChoiceRecord) -> Tuple[str, str]:
    try:
        metadata = json.loads(record.chosen_candidate_metadata or "{}")
    except json.JSONDecodeError:
        metadata = {}
    if not isinstance(metadata, dict):
        metadata = {}
    metadata.pop("raw_factor_object", None)
    normalised = {
        str(key): _normalise_text(str(value) if value is not None else "")
        for key, value in metadata.items()
    }
    return record.chosen_cf_value, json.dumps(normalised, ensure_ascii=True, sort_keys=True)


def _normalised_candidate_set_identity(
    candidates: Sequence[CharacterisationFactorCandidate],
) -> Tuple[Tuple[str, str], ...]:
    return tuple(sorted(_normalised_candidate_identity(candidate) for candidate in candidates))


def _normalised_record_candidate_set_identity(record: CFResolutionChoiceRecord) -> Tuple[Tuple[str, str], ...]:
    try:
        metadata_rows = json.loads(record.all_candidate_metadata or "[]")
    except json.JSONDecodeError:
        return ()
    if not isinstance(metadata_rows, list):
        return ()
    identities: list[Tuple[str, str]] = []
    for row in metadata_rows:
        if not isinstance(row, dict):
            return ()
        metadata = dict(row)
        metadata.pop("raw_factor_object", None)
        cf_value = metadata.get("cf_value", "")
        try:
            normalised_cf_value = _normalise_float(float(cf_value))
        except (TypeError, ValueError):
            normalised_cf_value = _normalise_text(str(cf_value) if cf_value is not None else "")
        normalised = {
            str(key): _normalise_text(str(value) if value is not None else "")
            for key, value in metadata.items()
        }
        identities.append((normalised_cf_value, json.dumps(normalised, ensure_ascii=True, sort_keys=True)))
    return tuple(sorted(identities))


def _record_from_row(row: Dict[str, str]) -> CFResolutionChoiceRecord:
    return CFResolutionChoiceRecord(
        category_id=str(row.get("category_id") or "").strip(),
        category_name=str(row.get("category_name") or "").strip(),
        method_id=str(row.get("method_id") or "").strip(),
        method_name=str(row.get("method_name") or "").strip(),
        flow_id=str(row.get("flow_id") or "").strip(),
        flow_name=str(row.get("flow_name") or "").strip(),
        process_id=str(row.get("process_id") or "").strip(),
        process_name=str(row.get("process_name") or "").strip(),
        exchange_id=str(row.get("exchange_id") or "").strip(),
        exchange_index=str(row.get("exchange_index") or "").strip(),
        chosen_cf_value=str(row.get("chosen_cf_value") or "").strip(),
        chosen_candidate_metadata=str(row.get("chosen_candidate_metadata") or "").strip(),
        rejected_cf_values=str(row.get("rejected_cf_values") or "").strip(),
        rejected_candidate_metadata=str(row.get("rejected_candidate_metadata") or "").strip(),
        all_candidate_cf_values=str(row.get("all_candidate_cf_values") or "").strip(),
        all_candidate_metadata=str(row.get("all_candidate_metadata") or "").strip(),
        choice_origin=str(row.get("choice_origin") or "").strip(),
        timestamp=str(row.get("timestamp") or "").strip(),
    )


def load_cf_resolution_choice_history(path: str | Path | None) -> list[CFResolutionChoiceRecord]:
    if not path:
        return []
    file_path = Path(path)
    if not file_path.exists():
        return []
    rows: list[CFResolutionChoiceRecord] = []
    with file_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            record = _record_from_row(row)
            if not record.category_id or not record.flow_id:
                continue
            rows.append(record)
    return rows


def load_cf_resolution_choices(path: str | Path | None) -> Dict[Tuple[str, str, str], CFResolutionChoiceRecord]:
    choices: Dict[Tuple[str, str, str], CFResolutionChoiceRecord] = {}
    for record in load_cf_resolution_choice_history(path):
        choices[(record.method_id, record.category_id, record.flow_id)] = record
    return choices


def write_cf_resolution_choices(
    path: str | Path,
    records: Sequence[CFResolutionChoiceRecord],
) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CHOICE_FILE_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow({field: getattr(record, field, "") for field in CHOICE_FILE_COLUMNS})


def _split_candidates(
    candidates: Sequence[CharacterisationFactorCandidate],
    chosen_candidate: CharacterisationFactorCandidate,
) -> tuple[list[CharacterisationFactorCandidate], list[CharacterisationFactorCandidate]]:
    chosen: list[CharacterisationFactorCandidate] = []
    rejected: list[CharacterisationFactorCandidate] = []
    chosen_identity = _normalised_candidate_identity(chosen_candidate)
    selected_once = False
    for candidate in candidates:
        if not selected_once and _normalised_candidate_identity(candidate) == chosen_identity:
            chosen.append(candidate)
            selected_once = True
        else:
            rejected.append(candidate)
    if not chosen:
        chosen = [chosen_candidate]
    return chosen, rejected


def _make_choice_record(
    *,
    context: CFAmbiguityContext,
    chosen_candidate: CharacterisationFactorCandidate,
    candidates: Sequence[CharacterisationFactorCandidate],
    choice_origin: str,
    timestamp: str,
) -> CFResolutionChoiceRecord:
    chosen, rejected = _split_candidates(candidates, chosen_candidate)
    return CFResolutionChoiceRecord(
        category_id=context.category_id,
        category_name=context.category_name,
        method_id=context.method_id,
        method_name=context.method_name,
        flow_id=context.flow_id,
        flow_name=context.flow_name,
        process_id=context.process_id,
        process_name=context.process_name,
        exchange_id=context.exchange_id,
        exchange_index=str(context.exchange_index),
        chosen_cf_value=_normalise_float(chosen_candidate.cf_value),
        chosen_candidate_metadata=candidate_metadata_json(chosen_candidate),
        rejected_cf_values=candidate_values_json(rejected),
        rejected_candidate_metadata=candidate_metadata_list_json(rejected),
        all_candidate_cf_values=candidate_values_json(candidates),
        all_candidate_metadata=candidate_metadata_list_json(candidates),
        choice_origin=choice_origin,
        timestamp=timestamp,
    )


class CFResolutionManager:
    def __init__(
        self,
        *,
        mode: str,
        choices_path: str | None,
        prompt: Optional[PromptCallback] = None,
    ) -> None:
        self.mode = mode
        self.choices_path = str(Path(choices_path)) if choices_path else ""
        self.prompt = prompt
        self.summary = CFResolutionSummary()
        self._choice_history = load_cf_resolution_choice_history(self.choices_path)
        self._choices = load_cf_resolution_choices(self.choices_path)
        self._ambiguity_keys_seen: set[str] = set()

    @property
    def choices(self) -> Dict[Tuple[str, str, str], CFResolutionChoiceRecord]:
        return self._choices

    @property
    def choice_history(self) -> Sequence[CFResolutionChoiceRecord]:
        return tuple(self._choice_history)

    def write_choices(self) -> None:
        if not self.choices_path:
            return
        write_cf_resolution_choices(self.choices_path, self._choice_history)

    def note_found(self, key: str) -> None:
        self.summary.n_cf_ambiguities_found += 1
        if key not in self._ambiguity_keys_seen:
            self._ambiguity_keys_seen.add(key)
            self.summary.n_cf_ambiguity_keys_unique = len(self._ambiguity_keys_seen)

    def record_automatic_resolution(self) -> None:
        self.summary.n_cf_ambiguities_resolved_automatically += 1

    def resolve_ambiguity(
        self,
        context: CFAmbiguityContext,
        candidates: Sequence[CharacterisationFactorCandidate],
    ) -> CFResolutionDecision:
        reused_record, reused_candidate = self._reuse_saved_choice(context, candidates)
        if reused_record is not None and reused_candidate is not None:
            timestamp = datetime.now(timezone.utc).isoformat()
            audit_record = _make_choice_record(
                context=context,
                chosen_candidate=reused_candidate,
                candidates=candidates,
                choice_origin="reused",
                timestamp=timestamp,
            )
            self._append_choice_record(audit_record)
            self.summary.n_cf_ambiguities_resolved_by_user_choice += 1
            self.summary.n_cf_resolution_choices_reused += 1
            return CFResolutionDecision(
                status="reused_choice",
                candidate=reused_candidate,
                reason="Reused a previously saved CF choice for the same ambiguity key.",
                audit_record=audit_record,
            )

        if self.mode == "gui" and self.prompt is not None:
            prompt_result = self.prompt(context, candidates)
            if prompt_result.action == "select":
                if prompt_result.candidate_index is None:
                    raise ValueError("CF ambiguity prompt returned select without a candidate index")
                if prompt_result.candidate_index < 0 or prompt_result.candidate_index >= len(candidates):
                    raise ValueError("CF ambiguity prompt returned an out-of-range candidate index")
                candidate = candidates[prompt_result.candidate_index]
                timestamp = datetime.now(timezone.utc).isoformat()
                audit_record = _make_choice_record(
                    context=context,
                    chosen_candidate=candidate,
                    candidates=candidates,
                    choice_origin="new",
                    timestamp=timestamp,
                )
                self._append_choice_record(audit_record)
                self.summary.n_cf_unique_user_decisions += 1
                self.summary.n_cf_ambiguities_resolved_by_user_choice += 1
                return CFResolutionDecision(
                    status="user_choice",
                    candidate=candidate,
                    reason="User explicitly selected a CF candidate in the GUI ambiguity dialog.",
                    audit_record=audit_record,
                )
            if prompt_result.action == "cancel_run":
                self.summary.n_cf_ambiguities_unresolved += 1
                return CFResolutionDecision(
                    status="cancel_run",
                    candidate=None,
                    reason="User cancelled the run from the CF ambiguity dialog.",
                )
            self.summary.n_cf_ambiguities_unresolved += 1
            return CFResolutionDecision(
                status="unresolved",
                candidate=None,
                reason="User chose Skip/Fail in the CF ambiguity dialog.",
            )

        self.summary.n_cf_ambiguities_unresolved += 1
        return CFResolutionDecision(
            status="unresolved",
            candidate=None,
            reason="No automatic or explicit CF resolution was available.",
        )

    def _reuse_saved_choice(
        self,
        context: CFAmbiguityContext,
        candidates: Sequence[CharacterisationFactorCandidate],
    ) -> tuple[Optional[CFResolutionChoiceRecord], Optional[CharacterisationFactorCandidate]]:
        record = self._choices.get((context.method_id, context.category_id, context.flow_id))
        if record is None:
            return None, None
        if _normalised_record_candidate_set_identity(record) != _normalised_candidate_set_identity(candidates):
            return None, None
        record_identity = _normalised_choice_identity(record)
        for candidate in candidates:
            if _normalised_candidate_identity(candidate) == record_identity:
                return record, candidate
        return None, None

    def _append_choice_record(self, record: CFResolutionChoiceRecord) -> None:
        self._choice_history.append(record)
        self._choices[(record.method_id, record.category_id, record.flow_id)] = record
        self.write_choices()
