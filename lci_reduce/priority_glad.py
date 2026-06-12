"""Direct priority-to-GLAD target matching for arbitrary source priority files."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from .errors import DataFormatError, LciReduceError
from .manifest import write_manifest_csv
from .priority_analyser import BASE_PRIORITY_COLUMNS, PriorityDataset, PriorityRecord, TauColumnPair, load_priority_dataset


_EPSILON = 1e-12
_GLAD_ASSET_DIR = Path(__file__).resolve().parent.parent / "priority_glad"
_SNAPSHOT_COLUMNS = [
    "source_file",
    "list_name",
    "flow_uuid",
    "flow_name",
    "context",
    "unit",
    "cas_number",
    "synonyms",
]
_TARGETS: dict[str, dict[str, str]] = {
    "ecoinventEFv3.7": {
        "display_name": "ecoinventEFv3.7",
        "snapshot": "ecoinventEFv3.7.csv",
        "upstream_file": "Mapping/Input/Flowlists/ecoinventEFv3.7.csv",
    },
    "ILCD_EFv3.0": {
        "display_name": "ILCD_EFv3.0",
        "snapshot": "ILCD_EFv3.0.csv",
        "upstream_file": "Mapping/Input/Flowlists/ILCD_EFv3.0.csv",
    },
    "FEDEFLv1.0.3": {
        "display_name": "FEDEFLv1.0.3",
        "snapshot": "FEDEFLv1.0.3.csv",
        "upstream_file": "Mapping/Input/Flowlists/FEDEFLv1.0.3.csv",
    },
    "IDEA_EFv2.3": {
        "display_name": "IDEA_EFv2.3",
        "snapshot": "IDEA_EFv2.3.csv",
        "upstream_file": "Mapping/Input/Flowlists/IDEA_EFv2.3.csv",
    },
}
_FULL_STATUS_ORDER = [
    "exact_consensus",
    "exact_uuid",
    "exact_name_context_unit",
    "conflict",
    "ambiguous",
    "unmatched",
]
_UNRESOLVED_STATUSES = {"conflict", "ambiguous", "unmatched"}
_MATCHED_STATUSES = {"exact_consensus", "exact_uuid", "exact_name_context_unit"}


class PriorityGladError(LciReduceError):
    """Raised when direct GLAD matching cannot be completed safely."""


@dataclass(frozen=True)
class GladFlowRecord:
    source_file: str
    list_name: str
    flow_uuid: str
    flow_name: str
    context: str
    unit: str
    cas_number: str
    synonyms: str
    original_order: int


@dataclass(frozen=True)
class GladMatchResult:
    target_key: str
    target_list: str
    match_status: str
    match_rule: str
    matched_target_uuid: str
    matched_target_flow_name: str
    matched_target_context: str
    matched_target_unit: str
    candidate_count: int
    notes: str
    candidates: tuple[GladFlowRecord, ...]

    @property
    def is_matched(self) -> bool:
        return self.match_status in _MATCHED_STATUSES

    @property
    def is_unresolved(self) -> bool:
        return self.match_status in _UNRESOLVED_STATUSES


@dataclass(frozen=True)
class GladTargetDataset:
    target_key: str
    display_name: str
    source_path: str
    rows: tuple[GladFlowRecord, ...]
    rows_by_uuid: dict[str, tuple[GladFlowRecord, ...]]
    rows_by_name_context_unit: dict[str, tuple[GladFlowRecord, ...]]


@dataclass(frozen=True)
class PriorityGladRowResult:
    priority_row: PriorityRecord
    pair: TauColumnPair
    target_results: dict[str, GladMatchResult]


@dataclass(frozen=True)
class PriorityGladSummary:
    priority_csv: str
    metadata_json: str | None
    selected_audit_tau: float
    summary_json_path: str
    compact_summary_csv_path: str
    per_target_full_csv_paths: dict[str, str]
    per_target_unmatched_csv_paths: dict[str, str]
    total_rows: int
    target_status_counts: dict[str, dict[str, int]]
    matched_all_targets: int
    matched_no_targets: int
    matched_some_targets: int
    asset_manifest: dict[str, Any]


def normalise_text(value: str | None) -> str:
    return " ".join(str(value or "").strip().split())


def _normalise_context_parts(parts: Iterable[str]) -> str:
    cleaned = [normalise_text(part) for part in parts if normalise_text(part)]
    if not cleaned:
        return ""
    root_labels = {
        "elementary flows",
        "elementary flow",
        "emissions",
        "emission",
        "resources",
        "resource",
    }
    while cleaned and cleaned[0].casefold() in root_labels:
        cleaned = cleaned[1:]
    return "/".join(cleaned)


def normalise_context(value: str | None) -> str:
    return _normalise_context_parts(str(value or "").split("/"))


def normalise_priority_context(compartment: str | None, subcompartment: str | None) -> str:
    parts = [str(compartment or "").strip()]
    tail = str(subcompartment or "").strip()
    if tail:
        parts.extend(part.strip() for part in tail.split("/") if part.strip())
    return _normalise_context_parts(parts)


def build_name_context_unit_key(flow_name: str | None, context: str | None, unit: str | None) -> str:
    return "||".join(
        [
            normalise_text(flow_name),
            normalise_context(context),
            normalise_text(unit),
        ]
    )


def build_priority_name_context_unit_key(row: PriorityRecord) -> str:
    return "||".join(
        [
            normalise_text(row.flow_name),
            normalise_priority_context(row.compartment, row.subcompartment),
            normalise_text(row.reference_unit),
        ]
    )


def classify_unresolved_row(row: PriorityRecord, pair: TauColumnPair) -> str:
    eta_value = row.eta(pair)
    loss_value = row.loss_max(pair)
    if eta_value > _EPSILON:
        return "certificate_breaking"
    if loss_value > _EPSILON:
        return "group_risk_only"
    if row.characterised_occurrence_count <= 0:
        return "not_lcia_visible"
    return "low_priority_unmatched"


def glad_asset_dir() -> Path:
    return _GLAD_ASSET_DIR


def load_asset_manifest(asset_dir: str | Path | None = None) -> dict[str, Any]:
    base_dir = Path(asset_dir) if asset_dir is not None else glad_asset_dir()
    manifest_path = base_dir / "manifest.json"
    if not manifest_path.exists():
        raise PriorityGladError(f"GLAD asset manifest not found: {manifest_path}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PriorityGladError(f"GLAD asset manifest is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise PriorityGladError("GLAD asset manifest must contain a single JSON object.")
    return payload


@lru_cache(maxsize=None)
def _load_target_dataset_cached(target_key: str, asset_dir_text: str) -> GladTargetDataset:
    return _load_target_dataset_uncached(target_key, Path(asset_dir_text))


def load_target_dataset(target_key: str, asset_dir: str | Path | None = None) -> GladTargetDataset:
    base_dir = Path(asset_dir) if asset_dir is not None else glad_asset_dir()
    return _load_target_dataset_cached(target_key, str(base_dir.resolve()))


def _load_target_dataset_uncached(target_key: str, asset_dir: Path) -> GladTargetDataset:
    target = _TARGETS.get(target_key)
    if target is None:
        raise PriorityGladError(f"Unsupported GLAD target: {target_key}")
    snapshot_path = asset_dir / target["snapshot"]
    if not snapshot_path.exists():
        raise PriorityGladError(f"GLAD target snapshot not found: {snapshot_path}")
    with snapshot_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        missing = [column for column in _SNAPSHOT_COLUMNS if column not in fieldnames]
        if missing:
            raise PriorityGladError(
                f"GLAD snapshot {snapshot_path.name} is missing required columns: {', '.join(missing)}"
            )
        rows: list[GladFlowRecord] = []
        for index, raw_row in enumerate(reader):
            row = {field: normalise_text((raw_row or {}).get(field, "")) for field in fieldnames}
            rows.append(
                GladFlowRecord(
                    source_file=row["source_file"],
                    list_name=row["list_name"],
                    flow_uuid=row["flow_uuid"],
                    flow_name=row["flow_name"],
                    context=normalise_context(row["context"]),
                    unit=row["unit"],
                    cas_number=row.get("cas_number", ""),
                    synonyms=row.get("synonyms", ""),
                    original_order=index,
                )
            )
    rows_by_uuid: dict[str, list[GladFlowRecord]] = {}
    rows_by_key: dict[str, list[GladFlowRecord]] = {}
    for row in rows:
        if row.flow_uuid:
            rows_by_uuid.setdefault(row.flow_uuid, []).append(row)
        key = build_name_context_unit_key(row.flow_name, row.context, row.unit)
        rows_by_key.setdefault(key, []).append(row)
    return GladTargetDataset(
        target_key=target_key,
        display_name=target["display_name"],
        source_path=str(snapshot_path),
        rows=tuple(rows),
        rows_by_uuid={key: tuple(value) for key, value in rows_by_uuid.items()},
        rows_by_name_context_unit={key: tuple(value) for key, value in rows_by_key.items()},
    )


def match_priority_row_to_target(row: PriorityRecord, target_dataset: GladTargetDataset) -> GladMatchResult:
    uuid_value = normalise_text(row.flow_id)
    name_context_unit_key = build_priority_name_context_unit_key(row)
    uuid_matches = list(target_dataset.rows_by_uuid.get(uuid_value, ())) if uuid_value else []
    key_matches = list(target_dataset.rows_by_name_context_unit.get(name_context_unit_key, ()))
    union_candidates = _dedupe_candidates([*uuid_matches, *key_matches])
    if len(uuid_matches) > 1 or len(key_matches) > 1:
        return GladMatchResult(
            target_key=target_dataset.target_key,
            target_list=target_dataset.display_name,
            match_status="ambiguous",
            match_rule=_describe_rule(bool(uuid_matches), bool(key_matches)),
            matched_target_uuid="",
            matched_target_flow_name="",
            matched_target_context="",
            matched_target_unit="",
            candidate_count=len(union_candidates),
            notes=_ambiguity_note(uuid_matches, key_matches),
            candidates=tuple(union_candidates),
        )
    if len(uuid_matches) == 1 and len(key_matches) == 1:
        if uuid_matches[0] == key_matches[0]:
            return _build_match_result(target_dataset, "exact_consensus", "uuid+name_context_unit", uuid_matches[0], union_candidates, "")
        return GladMatchResult(
            target_key=target_dataset.target_key,
            target_list=target_dataset.display_name,
            match_status="conflict",
            match_rule="uuid_vs_name_context_unit",
            matched_target_uuid="",
            matched_target_flow_name="",
            matched_target_context="",
            matched_target_unit="",
            candidate_count=len(union_candidates),
            notes="UUID and Name+Context+Unit resolved uniquely to different GLAD target rows.",
            candidates=tuple(union_candidates),
        )
    if len(uuid_matches) == 1:
        return _build_match_result(target_dataset, "exact_uuid", "uuid", uuid_matches[0], union_candidates, "")
    if len(key_matches) == 1:
        return _build_match_result(target_dataset, "exact_name_context_unit", "name_context_unit", key_matches[0], union_candidates, "")
    return GladMatchResult(
        target_key=target_dataset.target_key,
        target_list=target_dataset.display_name,
        match_status="unmatched",
        match_rule="",
        matched_target_uuid="",
        matched_target_flow_name="",
        matched_target_context="",
        matched_target_unit="",
        candidate_count=0,
        notes="No exact UUID or exact normalized Name+Context+Unit match was found in the GLAD target list.",
        candidates=(),
    )


def build_priority_glad_rows(
    priority_csv: str | Path,
    metadata_json: str | Path | None = None,
    audit_tau: float | None = None,
    *,
    asset_dir: str | Path | None = None,
) -> tuple[PriorityDataset, TauColumnPair, list[PriorityGladRowResult], dict[str, Any]]:
    dataset = load_priority_dataset(priority_csv, metadata_json)
    pair = dataset.get_tau_pair(audit_tau)
    manifest = load_asset_manifest(asset_dir)
    rows: list[PriorityGladRowResult] = []
    for row in dataset.rows:
        target_results = {
            target_key: match_priority_row_to_target(row, load_target_dataset(target_key, asset_dir))
            for target_key in _TARGETS
        }
        rows.append(PriorityGladRowResult(priority_row=row, pair=pair, target_results=target_results))
    return dataset, pair, rows, manifest


def write_priority_glad_outputs(
    priority_csv: str | Path,
    output_dir: str | Path,
    metadata_json: str | Path | None = None,
    audit_tau: float | None = None,
    *,
    asset_dir: str | Path | None = None,
) -> PriorityGladSummary:
    dataset, pair, rows, asset_manifest = build_priority_glad_rows(
        priority_csv,
        metadata_json,
        audit_tau,
        asset_dir=asset_dir,
    )
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    per_target_full_csv_paths: dict[str, str] = {}
    per_target_unmatched_csv_paths: dict[str, str] = {}
    target_status_counts: dict[str, dict[str, int]] = {}
    target_top_unresolved: dict[str, list[dict[str, Any]]] = {}
    for target_key, target_info in _TARGETS.items():
        full_rows = [serialise_priority_glad_row(item, item.target_results[target_key]) for item in rows]
        full_path = output_root / f"{target_key}_matches.csv"
        write_manifest_csv(full_path, full_rows, priority_glad_output_columns(pair))
        per_target_full_csv_paths[target_key] = str(full_path)

        unresolved_items = [
            item
            for item in rows
            if item.target_results[target_key].is_unresolved
        ]
        unresolved_items.sort(key=lambda item: _unresolved_sort_key(item.priority_row, pair))
        unresolved_rows = [serialise_priority_glad_row(item, item.target_results[target_key]) for item in unresolved_items]
        unresolved_path = output_root / f"{target_key}_unmatched.csv"
        write_manifest_csv(unresolved_path, unresolved_rows, priority_glad_output_columns(pair))
        per_target_unmatched_csv_paths[target_key] = str(unresolved_path)

        status_counts = {status: 0 for status in _FULL_STATUS_ORDER}
        for item in rows:
            status_counts[item.target_results[target_key].match_status] += 1
        target_status_counts[target_key] = status_counts
        target_top_unresolved[target_key] = unresolved_rows[:20]

    matched_all_targets = sum(
        1 for item in rows if all(result.is_matched for result in item.target_results.values())
    )
    matched_no_targets = sum(
        1 for item in rows if not any(result.is_matched for result in item.target_results.values())
    )
    matched_some_targets = len(rows) - matched_all_targets - matched_no_targets

    summary_payload = {
        "priority_csv": str(priority_csv),
        "metadata_json": str(metadata_json) if metadata_json else "",
        "selected_audit_tau": pair.tau,
        "eta_column": pair.eta_column,
        "loss_max_column": pair.loss_max_column,
        "total_rows": len(rows),
        "targets": list(_TARGETS),
        "target_status_counts": target_status_counts,
        "matched_all_targets": matched_all_targets,
        "matched_no_targets": matched_no_targets,
        "matched_some_targets": matched_some_targets,
        "top_unresolved_by_target": target_top_unresolved,
        "asset_manifest": asset_manifest,
        "per_target_full_csv_paths": per_target_full_csv_paths,
        "per_target_unmatched_csv_paths": per_target_unmatched_csv_paths,
    }
    summary_json_path = output_root / "priority_glad_summary.json"
    summary_json_path.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    compact_summary_rows = [
        {
            "target_list": target_key,
            "total_rows": len(rows),
            **{status: target_status_counts[target_key][status] for status in _FULL_STATUS_ORDER},
            "matched_rows": sum(target_status_counts[target_key][status] for status in _MATCHED_STATUSES),
            "unresolved_rows": sum(target_status_counts[target_key][status] for status in _UNRESOLVED_STATUSES),
        }
        for target_key in _TARGETS
    ]
    compact_summary_csv_path = output_root / "priority_glad_summary.csv"
    write_manifest_csv(
        compact_summary_csv_path,
        compact_summary_rows,
        ["target_list", "total_rows", *_FULL_STATUS_ORDER, "matched_rows", "unresolved_rows"],
    )

    return PriorityGladSummary(
        priority_csv=str(priority_csv),
        metadata_json=str(metadata_json) if metadata_json else None,
        selected_audit_tau=pair.tau,
        summary_json_path=str(summary_json_path),
        compact_summary_csv_path=str(compact_summary_csv_path),
        per_target_full_csv_paths=per_target_full_csv_paths,
        per_target_unmatched_csv_paths=per_target_unmatched_csv_paths,
        total_rows=len(rows),
        target_status_counts=target_status_counts,
        matched_all_targets=matched_all_targets,
        matched_no_targets=matched_no_targets,
        matched_some_targets=matched_some_targets,
        asset_manifest=asset_manifest,
    )


def format_priority_glad_report(summary: PriorityGladSummary) -> str:
    lines = [
        "Priority -> GLAD target matching",
        f"Priority CSV: {summary.priority_csv}",
        f"Metadata JSON: {summary.metadata_json or '-'}",
        f"Selected tau: {format(float(summary.selected_audit_tau), '.15g')}",
        f"Total rows: {summary.total_rows}",
        f"Matched in all four targets: {summary.matched_all_targets}",
        f"Matched in no targets: {summary.matched_no_targets}",
        f"Matched in some but not all targets: {summary.matched_some_targets}",
        "",
        "Per-target status counts",
    ]
    for target_key in _TARGETS:
        counts = summary.target_status_counts[target_key]
        lines.append(
            f"  {target_key}: "
            + " | ".join(f"{status}={counts[status]}" for status in _FULL_STATUS_ORDER)
        )
    lines.extend(
        [
            "",
            f"Summary JSON: {summary.summary_json_path}",
            f"Summary CSV: {summary.compact_summary_csv_path}",
        ]
    )
    for target_key in _TARGETS:
        lines.append(f"  {target_key} full CSV: {summary.per_target_full_csv_paths[target_key]}")
        lines.append(f"  {target_key} unmatched CSV: {summary.per_target_unmatched_csv_paths[target_key]}")
    return "\n".join(lines)


def serialise_priority_glad_row(item: PriorityGladRowResult, match_result: GladMatchResult) -> dict[str, Any]:
    row = item.priority_row
    payload = {column: row.raw.get(column, "") for column in BASE_PRIORITY_COLUMNS}
    payload[item.pair.eta_column] = row.raw.get(item.pair.eta_column, "")
    payload[item.pair.loss_max_column] = row.raw.get(item.pair.loss_max_column, "")
    payload.update(
        {
            "selected_audit_tau": item.pair.tau_label,
            "eta_tau": row.eta(item.pair),
            "loss_max_tau": row.loss_max(item.pair),
            "target_list": match_result.target_list,
            "match_status": match_result.match_status,
            "match_rule": match_result.match_rule,
            "matched_target_uuid": match_result.matched_target_uuid,
            "matched_target_flow_name": match_result.matched_target_flow_name,
            "matched_target_context": match_result.matched_target_context,
            "matched_target_unit": match_result.matched_target_unit,
            "candidate_count": match_result.candidate_count,
            "notes": match_result.notes,
            "unresolved_class": classify_unresolved_row(row, item.pair) if match_result.is_unresolved else "",
            "candidate_rows": " || ".join(_candidate_text(candidate) for candidate in match_result.candidates),
        }
    )
    return payload


def priority_glad_output_columns(pair: TauColumnPair) -> list[str]:
    return [
        *BASE_PRIORITY_COLUMNS,
        pair.eta_column,
        pair.loss_max_column,
        "selected_audit_tau",
        "eta_tau",
        "loss_max_tau",
        "target_list",
        "match_status",
        "match_rule",
        "matched_target_uuid",
        "matched_target_flow_name",
        "matched_target_context",
        "matched_target_unit",
        "candidate_count",
        "notes",
        "unresolved_class",
        "candidate_rows",
    ]


def priority_glad_target_keys() -> tuple[str, ...]:
    return tuple(_TARGETS)


def _build_match_result(
    target_dataset: GladTargetDataset,
    status: str,
    rule: str,
    candidate: GladFlowRecord,
    union_candidates: list[GladFlowRecord],
    notes: str,
) -> GladMatchResult:
    return GladMatchResult(
        target_key=target_dataset.target_key,
        target_list=target_dataset.display_name,
        match_status=status,
        match_rule=rule,
        matched_target_uuid=candidate.flow_uuid,
        matched_target_flow_name=candidate.flow_name,
        matched_target_context=candidate.context,
        matched_target_unit=candidate.unit,
        candidate_count=len(union_candidates),
        notes=notes,
        candidates=tuple(union_candidates),
    )


def _candidate_text(candidate: GladFlowRecord) -> str:
    return " | ".join(
        part
        for part in [candidate.flow_uuid, candidate.flow_name, candidate.context, candidate.unit]
        if part
    )


def _dedupe_candidates(candidates: list[GladFlowRecord]) -> list[GladFlowRecord]:
    seen: set[tuple[str, str, str, str]] = set()
    result: list[GladFlowRecord] = []
    for candidate in candidates:
        key = (candidate.flow_uuid, candidate.flow_name, candidate.context, candidate.unit)
        if key in seen:
            continue
        seen.add(key)
        result.append(candidate)
    return result


def _describe_rule(has_uuid: bool, has_name_context_unit: bool) -> str:
    if has_uuid and has_name_context_unit:
        return "uuid+name_context_unit"
    if has_uuid:
        return "uuid"
    if has_name_context_unit:
        return "name_context_unit"
    return ""


def _ambiguity_note(uuid_matches: list[GladFlowRecord], key_matches: list[GladFlowRecord]) -> str:
    if len(uuid_matches) > 1 and len(key_matches) > 1:
        return "Both UUID and Name+Context+Unit produced multiple GLAD target candidates."
    if len(uuid_matches) > 1:
        return "UUID produced multiple GLAD target candidates."
    return "Name+Context+Unit produced multiple GLAD target candidates."


def _unresolved_sort_key(row: PriorityRecord, pair: TauColumnPair) -> tuple[Any, ...]:
    return (
        -row.eta(pair),
        -row.loss_max(pair),
        -row.characterised_occurrence_count,
        row.flow_id,
        row.original_order,
    )


def validate_glad_assets(asset_dir: str | Path | None = None) -> None:
    manifest = load_asset_manifest(asset_dir)
    targets = manifest.get("targets")
    if not isinstance(targets, dict):
        raise DataFormatError("GLAD asset manifest must contain a `targets` object.")
    for target_key in _TARGETS:
        load_target_dataset(target_key, asset_dir)
