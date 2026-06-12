"""LCIA flow-priority sidecar generation."""

from __future__ import annotations

import hashlib
import gc
import json
import statistics
import time
from array import array
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence

import numpy as np

from . import __version__
from .cf_resolution import CFResolutionManager
from .contribution import build_sparse_contribution_details, exchange_flow_id, exchange_unit_name, flow_compartment
from .ecospold1_reader import iter_ecospold_processes
from .errors import DataFormatError
from .archive_reader import index_archive, iter_source_entries, merge_unit_registries, parse_json_object
from .lcia import (
    collect_categories,
    ensure_category_factor_quality,
    impact_category_report,
    resolve_lcia_archives,
    select_lcia_categories,
)
from .manifest import write_manifest_csv
from .models import (
    CFAmbiguityRecord,
    CoverageRowMetadata,
    CreateProgressUpdate,
    FlowInfo,
    FlowPriorityConfig,
    FlowPriorityResult,
    ImpactCategory,
    UnitInfo,
    WarningRecord,
)
from .sparse_cover import (
    add_column_contribution,
    build_greedy_ladder_sparse,
    build_sparse_cover_matrix,
    column_nonzero_mask,
)


ProgressCallback = Callable[[CreateProgressUpdate], None]

_BASE_COLUMNS = [
    "flow_id",
    "flow_name",
    "compartment",
    "subcompartment",
    "reference_unit",
    "occurrence_count",
    "characterised_occurrence_count",
    "tau_entry_min",
    "tau_entry_median",
    "tau_entry_max",
]

_METADATA_RECORD_SAMPLE_LIMIT = 100


@dataclass
class GreedyLadder:
    order: List[int]
    entry_thresholds: np.ndarray
    lambda_after: np.ndarray
    full: np.ndarray
    active: np.ndarray


@dataclass
class _FlowAggregate:
    flow: FlowInfo
    reference_unit: str = ""
    occurrence_count: int = 0
    characterised_occurrence_count: int = 0
    tau_entries: array = field(default_factory=lambda: array("d"))
    loss_max_by_tau: Dict[float, float] = field(default_factory=dict)
    eta_by_tau: Dict[float, float] = field(default_factory=dict)
    eta_witness_by_tau: Dict[float, str] = field(default_factory=dict)


class _BoundedRecordSink:
    """Count records while keeping only a small in-memory metadata sample."""

    def __init__(self, *, sample_limit: int = _METADATA_RECORD_SAMPLE_LIMIT) -> None:
        self.count = 0
        self._sample_limit = sample_limit
        self._sample: list[Any] = []

    def append(self, record: Any) -> None:
        self.count += 1
        if len(self._sample) < self._sample_limit:
            self._sample.append(record)

    def __len__(self) -> int:
        return self.count

    def __iter__(self) -> Iterator[Any]:
        return iter(self._sample)

    @property
    def sample(self) -> list[Any]:
        return list(self._sample)


def _emit_progress(
    progress_callback: Optional[ProgressCallback],
    *,
    step: str,
    message: str,
    current: int,
    total: int,
    process_current: int | None = None,
    process_total: int | None = None,
    process_name: str = "",
) -> None:
    if progress_callback is None:
        return
    progress_callback(
        CreateProgressUpdate(
            step=step,
            message=message,
            current=current,
            total=total,
            stage_current=current,
            stage_total=total,
            process_current=process_current,
            process_total=process_total,
            process_name=process_name,
        )
    )


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _hash_folder(path: Path) -> str:
    digest = hashlib.sha256()
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        relative = file_path.relative_to(path).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        with file_path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _hash_path(path: str | None) -> str:
    if not path:
        return ""
    file_path = Path(path)
    if file_path.is_dir():
        return _hash_folder(file_path)
    return _hash_file(file_path)


def _normalise_tau_values(values: Sequence[float]) -> List[float]:
    if not values:
        raise ValueError("At least one audit tau value is required")
    normalised = sorted(float(value) for value in values)
    seen: set[str] = set()
    result: List[float] = []
    for value in normalised:
        if value <= 0 or value > 1:
            raise ValueError("Audit tau values must be in (0, 1]")
        key = format(value, ".15g")
        if key in seen:
            raise ValueError("Audit tau values must be unique")
        seen.add(key)
        result.append(value)
    return result


def _tau_suffix(tau: float) -> str:
    return format(float(tau), ".15g").replace(".", "_")


def _metric_columns(audit_tau_values: Sequence[float]) -> List[str]:
    columns: List[str] = []
    for tau in audit_tau_values:
        suffix = _tau_suffix(tau)
        columns.append(f"eta_{suffix}")
        columns.append(f"eta_{suffix}_witness")
        columns.append(f"loss_max_{suffix}")
    return columns


def _format_float(value: float | None) -> str:
    if value is None:
        return ""
    if not np.isfinite(value):
        return ""
    return format(float(value), ".15g")


def _flow_reference_unit(flow: FlowInfo, unit_registry: Dict[str, UnitInfo], fallback_unit: str) -> str:
    if flow.reference_flow_property_id:
        candidates = sorted(
            {
                unit.name
                for unit in unit_registry.values()
                if unit.is_reference_unit
                and unit.flow_property_id
                and str(unit.flow_property_id) == str(flow.reference_flow_property_id)
                and unit.name
            }
        )
        if candidates:
            return candidates[0]
    return fallback_unit


def _display_label(name: str | None, object_id: str | None) -> str:
    label = str(name or "").strip()
    if label:
        return label
    fallback = str(object_id or "").strip()
    if fallback:
        return fallback
    return "-"


def _archive_parse_warnings(archive: object | None) -> list[dict[str, str]]:
    if archive is None:
        return []
    extra = getattr(archive, "extra", {})
    if not isinstance(extra, dict):
        return []
    raw_warnings = extra.get("parse_warnings", [])
    warnings: list[dict[str, str]] = []
    if not isinstance(raw_warnings, list):
        return warnings
    for item in raw_warnings:
        if not isinstance(item, dict):
            continue
        source_file = str(item.get("source_file") or "").strip()
        message = str(item.get("message") or "").strip()
        error_type = str(item.get("error_type") or "").strip()
        if not source_file or not message:
            continue
        warnings.append(
            {
                "source_file": source_file,
                "message": message,
                "error_type": error_type,
            }
        )
    return warnings


def _parse_warning_record(
    *,
    source_name: str,
    source_file: str,
    message: str,
    input_role: str,
) -> WarningRecord:
    return WarningRecord(
        severity="warning",
        object_type="input_archive",
        object_id=source_name,
        object_name=source_name,
        process_id="",
        process_name="",
        flow_id="",
        flow_name="",
        category_id="",
        category_name="",
        message=f"Skipped malformed EcoSpold1 XML file in {input_role}: {source_file}. {message}",
        source_file=source_file,
    )


def _raise_if_no_selected_categories(
    *,
    selected_categories: Sequence[ImpactCategory],
    methods_input: str | None,
    input_parse_warnings: Sequence[dict[str, str]],
) -> None:
    if selected_categories:
        return
    message = "No usable LCIA categories were available for this priority run."
    if methods_input:
        message += " The external methods input did not provide any parseable impact categories."
    if input_parse_warnings:
        first_warning = input_parse_warnings[0]
        message += (
            f" {len(input_parse_warnings)} input XML file(s) were skipped due to parse errors. "
            f"First skipped file: {first_warning['source_file']}. "
            f"First error: {first_warning['message']}"
        )
    raise DataFormatError(message)


def _eta_witness(
    process_name: str | None,
    process_id: str | None,
    row_metadata: CoverageRowMetadata,
    sign: str,
) -> str:
    process_label = _display_label(process_name, process_id)
    category_label = _display_label(row_metadata.category_name, row_metadata.category_id)
    if row_metadata.scenario_type == "exact":
        return f"{process_label} | {category_label} | {sign}"
    scenario_label = row_metadata.scenario_label or row_metadata.scenario_id or "scenario"
    return f"{process_label} | {category_label} | {scenario_label} | {sign}"


def build_greedy_ladder(
    M: np.ndarray,
    exchange_keys: Sequence[str] | None = None,
    tol: float = 1e-12,
) -> GreedyLadder:
    array = np.asarray(M, dtype=float)
    if array.ndim != 2:
        raise ValueError("M must be 2D")
    if not np.isfinite(array).all():
        raise ValueError("M must be finite")
    if (array < 0).any():
        raise ValueError("Greedy ladders require non-negative contributions")
    n_categories, n_exchanges = array.shape
    keys = list(exchange_keys) if exchange_keys is not None else [f"{index:012d}" for index in range(n_exchanges)]
    if len(keys) != n_exchanges:
        raise ValueError("exchange_keys length must match number of exchanges")
    entry_thresholds = np.full(n_exchanges, np.nan, dtype=float)
    full = array.sum(axis=1)
    active = full > tol
    if n_categories == 0 or n_exchanges == 0 or not active.any():
        return GreedyLadder(order=[], entry_thresholds=entry_thresholds, lambda_after=np.zeros(0), full=full, active=active)
    weights = np.zeros(n_categories, dtype=float)
    weights[active] = 1.0 / full[active]
    selected = np.zeros(n_exchanges, dtype=bool)
    retained = np.zeros(n_categories, dtype=float)
    tie_order = np.lexsort((np.arange(n_exchanges), np.asarray(keys, dtype=object)))
    tie_rank = np.empty(n_exchanges, dtype=int)
    tie_rank[tie_order] = np.arange(n_exchanges)
    order: List[int] = []
    lambda_after: List[float] = []
    while True:
        remaining = np.maximum(full - retained, 0.0)
        scores = (np.minimum(array, remaining[:, None]) * weights[:, None]).sum(axis=0)
        scores[selected] = -np.inf
        best_score = float(scores.max(initial=-np.inf))
        if not np.isfinite(best_score) or best_score <= 0.0:
            break
        contenders = np.flatnonzero(~selected & np.isclose(scores, best_score, rtol=tol, atol=0.0))
        if contenders.size == 0:
            best_index = int(np.argmax(scores))
        else:
            best_index = int(contenders[np.argmin(tie_rank[contenders])])
        coverage_before = np.ones(n_categories, dtype=float)
        coverage_before[active] = np.clip(retained[active] / full[active], 0.0, 1.0)
        entry_thresholds[best_index] = float(coverage_before[active].min()) if active.any() else 1.0
        selected[best_index] = True
        order.append(best_index)
        retained += array[:, best_index]
        coverage_after = np.ones(n_categories, dtype=float)
        coverage_after[active] = np.clip(retained[active] / full[active], 0.0, 1.0)
        lambda_after.append(float(coverage_after[active].min()) if active.any() else 1.0)
    return GreedyLadder(
        order=order,
        entry_thresholds=entry_thresholds,
        lambda_after=np.asarray(lambda_after, dtype=float),
        full=full,
        active=active,
    )


def prefix_length_for_tau(ladder: GreedyLadder, tau: float, tol: float = 1e-12) -> int:
    if tau <= 0 or tau > 1:
        raise ValueError("tau must be in (0, 1]")
    if not ladder.order or not ladder.active.any():
        return 0
    for index, value in enumerate(ladder.lambda_after, start=1):
        if float(value) + tol >= tau:
            return index
    return len(ladder.order)


def single_flow_shortfall(tau: float, coverage_before_loss: float, loss: float) -> float:
    return max(float(tau) - (float(coverage_before_loss) - float(loss)), 0.0)


def _combine_entry_thresholds(
    positive: np.ndarray,
    negative: np.ndarray,
    nonzero_mask: np.ndarray,
) -> List[float | None]:
    combined: List[float | None] = []
    for index in range(nonzero_mask.size):
        if not nonzero_mask[index]:
            combined.append(None)
            continue
        values = [value for value in (positive[index], negative[index]) if np.isfinite(value)]
        combined.append(float(min(values)) if values else None)
    return combined


def _selected_methods(categories: Sequence[ImpactCategory]) -> List[dict]:
    rows: List[dict] = []
    seen: set[tuple[str, str, str]] = set()
    for category in categories:
        key = (str(category.method_id or ""), str(category.method_name or ""), str(category.method_path or ""))
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "method_id": category.method_id or "",
                "method_name": category.method_name or "",
                "method_path": category.method_path or "",
                "source_file": category.method_source_file or "",
            }
        )
    return rows


def _process_progress_message(process_index: int, process_total: int, process_name: str) -> str:
    label = process_name or "-"
    return f"Step 5/6: Process {process_index}/{process_total} | {label}"


def _ensure_flow_aggregate(
    aggregates: Dict[str, _FlowAggregate],
    flow: FlowInfo,
    *,
    fallback_unit: str,
    unit_registry: Dict[str, UnitInfo],
    audit_tau_values: Sequence[float],
) -> _FlowAggregate:
    aggregate = aggregates.get(flow.flow_id)
    if aggregate is None:
        aggregate = _FlowAggregate(
            flow=flow,
            reference_unit=_flow_reference_unit(flow, unit_registry, fallback_unit),
            loss_max_by_tau={tau: 0.0 for tau in audit_tau_values},
            eta_by_tau={tau: 0.0 for tau in audit_tau_values},
            eta_witness_by_tau={tau: "" for tau in audit_tau_values},
        )
        aggregates[flow.flow_id] = aggregate
    elif not aggregate.reference_unit and fallback_unit:
        aggregate.reference_unit = _flow_reference_unit(flow, unit_registry, fallback_unit)
    return aggregate


def _update_flow_metrics_for_sign(
    matrix: np.ndarray,
    ladder: GreedyLadder,
    flow_ids: Sequence[str],
    row_metadata: Sequence[CoverageRowMetadata],
    audit_tau_values: Sequence[float],
    aggregates: Dict[str, _FlowAggregate],
    tol: float,
    *,
    process_name: str,
    process_id: str,
    sign: str,
) -> None:
    if not ladder.active.any() or not ladder.order:
        return
    retained = np.zeros(matrix.shape[0], dtype=float)
    flow_contribs: Dict[str, np.ndarray] = {}
    cursor = 0
    for tau in audit_tau_values:
        target = prefix_length_for_tau(ladder, tau, tol=tol)
        while cursor < target:
            local_index = ladder.order[cursor]
            retained += matrix[:, local_index]
            flow_id = flow_ids[local_index]
            if flow_id not in flow_contribs:
                flow_contribs[flow_id] = np.zeros(matrix.shape[0], dtype=float)
            flow_contribs[flow_id] += matrix[:, local_index]
            cursor += 1
        coverage = np.ones(matrix.shape[0], dtype=float)
        coverage[ladder.active] = retained[ladder.active] / ladder.full[ladder.active]
        for flow_id, flow_vector in flow_contribs.items():
            losses = np.zeros(matrix.shape[0], dtype=float)
            losses[ladder.active] = flow_vector[ladder.active] / ladder.full[ladder.active]
            loss_max = float(losses[ladder.active].max(initial=0.0))
            if loss_max > aggregates[flow_id].loss_max_by_tau[tau]:
                aggregates[flow_id].loss_max_by_tau[tau] = loss_max
            shortfalls = np.maximum(tau - (coverage - losses), 0.0)
            shortfall = float(shortfalls[ladder.active].max(initial=0.0))
            if shortfall > aggregates[flow_id].eta_by_tau[tau]:
                aggregates[flow_id].eta_by_tau[tau] = shortfall
                active_indices = np.flatnonzero(ladder.active)
                witness_index = int(active_indices[np.argmax(shortfalls[ladder.active])])
                aggregates[flow_id].eta_witness_by_tau[tau] = _eta_witness(
                    process_name,
                    process_id,
                    row_metadata[witness_index],
                    sign,
                )


def _update_flow_metrics_for_sign_sparse(
    model: Any,
    ladder: GreedyLadder,
    flow_ids: Sequence[str],
    audit_tau_values: Sequence[float],
    aggregates: Dict[str, _FlowAggregate],
    tol: float,
    *,
    process_name: str,
    process_id: str,
    sign: str,
) -> None:
    if not ladder.active.any() or not ladder.order:
        return
    retained = np.zeros(model.n_rows, dtype=float)
    selected_columns_by_flow: Dict[str, List[int]] = {}
    cursor = 0
    active_indices = np.flatnonzero(ladder.active)
    full_active = ladder.full[ladder.active]
    for tau in audit_tau_values:
        target = prefix_length_for_tau(ladder, tau, tol=tol)
        while cursor < target:
            local_index = ladder.order[cursor]
            add_column_contribution(model, retained, local_index)
            selected_columns_by_flow.setdefault(flow_ids[local_index], []).append(local_index)
            cursor += 1
        coverage_active = retained[ladder.active] / full_active
        for flow_id, selected_columns in selected_columns_by_flow.items():
            flow_vector = np.zeros(model.n_rows, dtype=float)
            for local_index in selected_columns:
                add_column_contribution(model, flow_vector, local_index)
            flow_active = flow_vector[ladder.active] / full_active
            loss_max = float(flow_active.max(initial=0.0))
            if loss_max > aggregates[flow_id].loss_max_by_tau[tau]:
                aggregates[flow_id].loss_max_by_tau[tau] = loss_max
            shortfalls_active = np.maximum(tau - (coverage_active - flow_active), 0.0)
            shortfall = float(shortfalls_active.max(initial=0.0))
            if shortfall > aggregates[flow_id].eta_by_tau[tau]:
                aggregates[flow_id].eta_by_tau[tau] = shortfall
                witness_index = int(active_indices[np.argmax(shortfalls_active)])
                aggregates[flow_id].eta_witness_by_tau[tau] = _eta_witness(
                    process_name,
                    process_id,
                    model.row_metadata[witness_index],
                    sign,
                )


def _csv_row(
    aggregate: _FlowAggregate,
    audit_tau_values: Sequence[float],
) -> dict:
    tau_entries = aggregate.tau_entries
    compartment, subcompartment = flow_compartment(aggregate.flow)
    row = {
        "flow_id": aggregate.flow.flow_id,
        "flow_name": aggregate.flow.name,
        "compartment": compartment,
        "subcompartment": subcompartment,
        "reference_unit": aggregate.reference_unit,
        "occurrence_count": aggregate.occurrence_count,
        "characterised_occurrence_count": aggregate.characterised_occurrence_count,
        "tau_entry_min": _format_float(min(tau_entries) if tau_entries else None),
        "tau_entry_median": _format_float(statistics.median(tau_entries) if tau_entries else None),
        "tau_entry_max": _format_float(max(tau_entries) if tau_entries else None),
    }
    for tau in audit_tau_values:
        suffix = _tau_suffix(tau)
        row[f"eta_{suffix}"] = _format_float(aggregate.eta_by_tau[tau])
        row[f"eta_{suffix}_witness"] = aggregate.eta_witness_by_tau.get(tau, "") if aggregate.eta_by_tau[tau] > 0.0 else ""
        row[f"loss_max_{suffix}"] = _format_float(aggregate.loss_max_by_tau[tau])
    return row


def _category_path_warning(
    database_name: str,
    metadata_name: str,
    diagnostics: Any,
    *,
    source_format: str = "jsonld",
) -> WarningRecord | None:
    if diagnostics.n_unresolved_elementary_flows <= 0:
        return None
    source_label = "JSON-LD category resolution" if source_format == "jsonld" else "source-format flow parsing"
    return WarningRecord(
        severity="warning",
        object_type="database",
        object_id=database_name,
        object_name=database_name,
        process_id="",
        process_name="",
        flow_id="",
        flow_name="",
        category_id="",
        category_name="",
        message=(
            f"{diagnostics.n_unresolved_elementary_flows}/{diagnostics.n_elementary_flows} elementary flows "
            f"({format(diagnostics.pct_elementary_flows_with_category_path, '.2f')}% with category paths) "
            f"still have empty flow category paths after {source_label}. "
            "Their `compartment` and `subcompartment` CSV fields remain blank. "
            f"See `flow_category_path_diagnostics` in {metadata_name}."
        ),
        source_file=metadata_name,
    )


def create_flow_priority(
    config: FlowPriorityConfig,
    *,
    progress_callback: Optional[ProgressCallback] = None,
) -> FlowPriorityResult:
    audit_tau_values = _normalise_tau_values(config.audit_tau_values)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "lcia_flow_priority.csv"
    metadata_path = output_dir / "lcia_flow_priority_metadata.json"
    warnings = _BoundedRecordSink()
    cf_ambiguities = _BoundedRecordSink()
    selected_categories: List[ImpactCategory] = []
    empty_selected_categories: List[ImpactCategory] = []
    flow_aggregates: Dict[str, _FlowAggregate] = {}
    resolution_manager = CFResolutionManager(mode="cli")
    total_steps = 6
    current_step = 0
    _emit_progress(
        progress_callback,
        step="load_database",
        message="Step 1/6: Scanning database archive...",
        current=current_step,
        total=total_steps,
    )
    database_archive = index_archive(config.database, require_processes=True, require_flows=True)
    current_step += 1
    _emit_progress(
        progress_callback,
        step="load_database",
        message=(
            f"Step 1/6: Indexed {len(database_archive.processes)} processes and "
            f"{len(database_archive.flows)} flows."
        ),
        current=current_step,
        total=total_steps,
    )
    category_path_warning = _category_path_warning(
        database_archive.source_name,
        metadata_path.name,
        database_archive.category_path_diagnostics,
        source_format=database_archive.source_format,
    )
    if category_path_warning is not None:
        warnings.append(category_path_warning)
    for parse_warning in _archive_parse_warnings(database_archive):
        warnings.append(
            _parse_warning_record(
                source_name=database_archive.source_name,
                source_file=parse_warning["source_file"],
                message=parse_warning["message"],
                input_role="database input",
            )
        )

    _emit_progress(
        progress_callback,
        step="load_methods",
        message="Step 2/6: Scanning optional methods input...",
        current=current_step,
        total=total_steps,
    )
    methods_archive = (
        index_archive(
            config.methods,
            require_processes=False,
            require_flows=False,
            ecospold_xml_error_policy="warn_skip",
        )
        if config.methods
        else None
    )
    method_parse_warnings = _archive_parse_warnings(methods_archive)
    if methods_archive is not None:
        source_name = Path(config.methods).name if config.methods else methods_archive.source_name
        for parse_warning in method_parse_warnings:
            warnings.append(
                _parse_warning_record(
                    source_name=source_name,
                    source_file=parse_warning["source_file"],
                    message=parse_warning["message"],
                    input_role="optional methods input",
                )
            )
    current_step += 1
    _emit_progress(
        progress_callback,
        step="load_methods",
        message=(
            (
                "Step 2/6: Optional methods indexed."
                if not method_parse_warnings
                else (
                    "Step 2/6: Optional methods indexed. "
                    f"Skipped {len(method_parse_warnings)} malformed EcoSpold1 XML file(s) in the optional methods input."
                )
            )
            if methods_archive is not None
            else "Step 2/6: No external methods input provided."
        ),
        current=current_step,
        total=total_steps,
    )
    input_parse_warnings = [*_archive_parse_warnings(database_archive), *method_parse_warnings]

    _emit_progress(
        progress_callback,
        step="collect_categories",
        message="Step 3/6: Collecting LCIA categories and CF candidates...",
        current=current_step,
        total=total_steps,
    )
    archives, lcia_method_source, internal_lcia_methods_ignored = resolve_lcia_archives(
        database_archive,
        methods_archive,
    )
    active_unit_registry = merge_unit_registries(
        getattr(database_archive, "units", {}),
        getattr(methods_archive, "units", {}) if methods_archive is not None else {},
    )
    categories = collect_categories(
        archives,
        ambiguity_records=cf_ambiguities,
        diagnostic_file="lcia_flow_priority_metadata.json",
    )
    current_step += 1
    _emit_progress(
        progress_callback,
        step="collect_categories",
        message=f"Step 3/6: Collected {len(categories)} LCIA categories.",
        current=current_step,
        total=total_steps,
    )

    _emit_progress(
        progress_callback,
        step="select_categories",
        message="Step 4/6: Selecting LCIA categories for this audit...",
        current=current_step,
        total=total_steps,
    )
    selected_categories = list(select_lcia_categories(categories, config.method_selection))
    empty_selected_categories = list(ensure_category_factor_quality(selected_categories, warnings))
    _raise_if_no_selected_categories(
        selected_categories=selected_categories,
        methods_input=config.methods,
        input_parse_warnings=input_parse_warnings,
    )
    current_step += 1
    _emit_progress(
        progress_callback,
        step="select_categories",
        message=f"Step 4/6: Selected {len(selected_categories)} LCIA categories.",
        current=current_step,
        total=total_steps,
    )

    process_total = len(database_archive.processes)
    _emit_progress(
        progress_callback,
        step="audit_processes",
        message=f"Step 5/6: Auditing {process_total} processes...",
        current=current_step,
        total=total_steps,
        process_current=0,
        process_total=process_total,
    )
    last_progress_at = 0.0
    process_index = 0
    if database_archive.source_format == "ecospold1":
        process_iterable = iter_ecospold_processes(database_archive)
    else:
        process_lookup = {locator.path: locator for locator in database_archive.processes.values()}
        process_iterable = (
            (locator, parse_json_object(raw_bytes, rel_path))
            for rel_path, raw_bytes in iter_source_entries(database_archive.resolved_source_path)
            for locator in [process_lookup.get(rel_path)]
            if locator is not None
        )
    for locator, process_data in process_iterable:
        exchanges = list(process_data.get("exchanges") or [])
        sparse_details = build_sparse_contribution_details(
            exchanges=exchanges,
            flow_lookup=database_archive.flows,
            categories=selected_categories,
            unit_registry=active_unit_registry,
            strict_units=config.strict_units,
            tol=config.tolerance,
            allow_water_mass_volume_override=config.allow_water_mass_volume_override,
            process_data=process_data,
            warning_records=warnings,
            ambiguity_records=cf_ambiguities,
            diagnostic_file="lcia_flow_priority_metadata.json",
            resolution_manager=resolution_manager,
            max_scenario_rows_per_process=config.max_scenario_rows_per_process,
            max_candidate_set_size=config.max_candidate_set_size,
            include_resolved_mask=False,
        )
        candidate_indices = list(sparse_details.candidate_indices)
        exchange_keys = list(sparse_details.exchange_keys)
        positive_model = build_sparse_cover_matrix(
            sparse_details.rowsets,
            len(candidate_indices),
            positive=True,
            tol=config.tolerance,
        )
        negative_model = build_sparse_cover_matrix(
            sparse_details.rowsets,
            len(candidate_indices),
            positive=False,
            tol=config.tolerance,
        )
        flow_ids: List[str] = []
        characterised_mask = np.asarray(sparse_details.characterised_flags, dtype=bool)
        nonzero_characterised = column_nonzero_mask(positive_model) | column_nonzero_mask(negative_model)
        for local_index, exchange_index in enumerate(candidate_indices):
            exchange = exchanges[exchange_index]
            flow_id = exchange_flow_id(exchange)
            if not flow_id:
                continue
            flow = database_archive.flows[flow_id]
            aggregate = _ensure_flow_aggregate(
                flow_aggregates,
                flow,
                fallback_unit=exchange_unit_name(exchange) or "",
                unit_registry=active_unit_registry,
                audit_tau_values=audit_tau_values,
            )
            aggregate.occurrence_count += 1
            if characterised_mask[local_index]:
                aggregate.characterised_occurrence_count += 1
            flow_ids.append(flow_id)

        positive_sparse_ladder = build_greedy_ladder_sparse(
            positive_model,
            exchange_keys=exchange_keys,
            tol=config.tolerance,
        )
        negative_sparse_ladder = build_greedy_ladder_sparse(
            negative_model,
            exchange_keys=exchange_keys,
            tol=config.tolerance,
        )
        positive_ladder = GreedyLadder(
            order=list(positive_sparse_ladder.order),
            entry_thresholds=positive_sparse_ladder.entry_thresholds,
            lambda_after=positive_sparse_ladder.lambda_after,
            full=positive_sparse_ladder.full,
            active=positive_sparse_ladder.active,
        )
        negative_ladder = GreedyLadder(
            order=list(negative_sparse_ladder.order),
            entry_thresholds=negative_sparse_ladder.entry_thresholds,
            lambda_after=negative_sparse_ladder.lambda_after,
            full=negative_sparse_ladder.full,
            active=negative_sparse_ladder.active,
        )
        combined_entry = _combine_entry_thresholds(
            positive_ladder.entry_thresholds,
            negative_ladder.entry_thresholds,
            nonzero_characterised,
        )
        for local_index, tau_entry in enumerate(combined_entry):
            if tau_entry is None:
                continue
            flow_id = flow_ids[local_index]
            flow_aggregates[flow_id].tau_entries.append(tau_entry)

        _update_flow_metrics_for_sign_sparse(
            positive_model,
            positive_ladder,
            flow_ids,
            audit_tau_values,
            flow_aggregates,
            config.tolerance,
            process_name=str(process_data.get("name") or locator.name),
            process_id=str(process_data.get("@id") or process_data.get("id") or locator.object_id),
            sign="+",
        )
        _update_flow_metrics_for_sign_sparse(
            negative_model,
            negative_ladder,
            flow_ids,
            audit_tau_values,
            flow_aggregates,
            config.tolerance,
            process_name=str(process_data.get("name") or locator.name),
            process_id=str(process_data.get("@id") or process_data.get("id") or locator.object_id),
            sign="-",
        )

        process_index += 1
        process_label = str(process_data.get("name") or locator.name)
        if process_index % 50 == 0:
            gc.collect()
        now = time.monotonic()
        if process_index == 1 or process_index == process_total or now - last_progress_at >= 0.15:
            last_progress_at = now
            _emit_progress(
                progress_callback,
                step="audit_processes",
                message=_process_progress_message(process_index, process_total, process_label),
                current=current_step,
                total=total_steps,
                process_current=process_index,
                process_total=process_total,
                process_name=process_label,
            )
    current_step += 1
    _emit_progress(
        progress_callback,
        step="audit_processes",
        message=f"Step 5/6: Audited {process_total} processes.",
        current=current_step,
        total=total_steps,
        process_current=process_total,
        process_total=process_total,
    )

    _emit_progress(
        progress_callback,
        step="write_outputs",
        message="Step 6/6: Writing flow-priority outputs...",
        current=current_step,
        total=total_steps,
    )
    metric_columns = _metric_columns(audit_tau_values)
    fieldnames = [*_BASE_COLUMNS, *metric_columns]
    rows = (
        _csv_row(flow_aggregates[flow_id], audit_tau_values)
        for flow_id in sorted(flow_aggregates)
    )
    write_manifest_csv(csv_path, rows, fieldnames)
    metadata = {
        "file_type": "lcia_flow_priority",
        "schema_version": 1,
        "source_format": database_archive.source_format,
        "database_name": database_archive.source_name,
        "database_version": None,
        "database_hash": _hash_path(config.database),
        "lcia_source_hash": _hash_path(config.methods) if config.methods else "",
        "lcia_method_source": lcia_method_source,
        "internal_lcia_methods_ignored": internal_lcia_methods_ignored,
        "selected_methods": _selected_methods(selected_categories),
        "selected_impact_categories": [impact_category_report(category).__dict__ for category in selected_categories],
        "empty_selected_impact_categories": [
            impact_category_report(category).__dict__ for category in empty_selected_categories
        ],
        "audit_tau_values": audit_tau_values,
        "unit_policy": "strict" if config.strict_units else "non_strict",
        "allow_water_mass_volume_override": config.allow_water_mass_volume_override,
        "cf_resolution_policy": {
            "mode": "finite_scenario_rows",
            "cf_ambiguity_policy": "finite_scenario_rows",
            "scenario_limits": {
                "max_scenario_rows_per_process": config.max_scenario_rows_per_process,
                "max_candidate_set_size": config.max_candidate_set_size,
            },
        },
        "algorithm": "nested_greedy_lcia_flow_priority",
        "algorithm_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "csv_filename": csv_path.name,
        "csv_columns": fieldnames,
        "n_processes_total": process_total,
        "n_elementary_occurrences_total": sum(item.occurrence_count for item in flow_aggregates.values()),
        "n_flows_ranked": len(flow_aggregates),
        "flow_category_path_diagnostics": database_archive.category_path_diagnostics.__dict__,
        "n_warning_records": len(warnings),
        "n_cf_ambiguity_records": len(cf_ambiguities),
        "metadata_record_sample_limit": _METADATA_RECORD_SAMPLE_LIMIT,
        "warnings_truncated_in_metadata": len(warnings) > _METADATA_RECORD_SAMPLE_LIMIT,
        "cf_ambiguities_truncated_in_metadata": len(cf_ambiguities) > _METADATA_RECORD_SAMPLE_LIMIT,
        "diagnostics_policy": {
            "priority_default_outputs": [
                "lcia_flow_priority.csv",
                "lcia_flow_priority_metadata.json",
            ],
            "full_warning_records": "not_written_by_priority_default",
            "full_cf_ambiguity_records": "not_written_by_priority_default",
            "full_cf_ambiguity_review": "use_explore-ambiguities_workflow",
        },
        "cf_resolution_summary": resolution_manager.summary.__dict__,
        "notes": [
            "eta is the single-flow certificate shortfall after overshoot margin is subtracted.",
            "loss_max is the maximum raw retained-coverage loss caused by omitting one flow.",
            "Group eta cannot be computed exactly by summing single-flow eta values.",
            "Summed loss_max values provide a conservative upper bound for group-risk screening.",
        ],
        "warnings": [warning.__dict__ for warning in warnings.sample],
        "cf_ambiguity_samples": [record.__dict__ for record in cf_ambiguities.sample],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=True), encoding="utf-8")
    current_step += 1
    _emit_progress(
        progress_callback,
        step="write_outputs",
        message="Step 6/6: Flow-priority files ready.",
        current=current_step,
        total=total_steps,
        process_current=process_total,
        process_total=process_total,
    )
    return FlowPriorityResult(
        flow_priority_csv=str(csv_path),
        flow_priority_metadata_json=str(metadata_path),
        metadata=metadata,
    )
