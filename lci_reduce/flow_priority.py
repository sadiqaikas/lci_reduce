"""LCIA flow-priority sidecar generation."""

from __future__ import annotations

import hashlib
import json
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from . import __version__
from .cf_resolution import CFResolutionManager, PromptCallback
from .contribution import build_contribution_details, exchange_flow_id, exchange_unit_name, flow_compartment
from .jsonld_reader import index_archive, iter_source_entries, parse_json_object
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
    CreateProgressUpdate,
    FlowInfo,
    FlowPriorityConfig,
    FlowPriorityResult,
    ImpactCategory,
    UnitInfo,
    WarningRecord,
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
    tau_entries: List[float] = field(default_factory=list)
    resolved_categories: set[str] = field(default_factory=set)
    loss_max_by_tau: Dict[float, float] = field(default_factory=dict)
    eta_by_tau: Dict[float, float] = field(default_factory=dict)
    eta_witness_by_tau: Dict[float, str] = field(default_factory=dict)


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


def _eta_witness(process_name: str | None, process_id: str | None, category: ImpactCategory, sign: str) -> str:
    process_label = _display_label(process_name, process_id)
    category_label = _display_label(category.name, category.category_id)
    return f"{process_label} | {category_label} | {sign}"


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
    active = full > 0.0
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


def _choice_history_hash(resolution_manager: CFResolutionManager) -> str:
    if not resolution_manager.choice_history:
        return ""
    payload = json.dumps([record.__dict__ for record in resolution_manager.choice_history], ensure_ascii=True, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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
    categories: Sequence[ImpactCategory],
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
                    categories[witness_index],
                    sign,
                )


def _csv_row(
    aggregate: _FlowAggregate,
    audit_tau_values: Sequence[float],
    selected_category_ids: set[str],
) -> dict:
    tau_entries = aggregate.tau_entries
    compartment, subcompartment = flow_compartment(aggregate.flow)
    if not aggregate.resolved_categories:
        cf_status = "uncharacterised"
    elif aggregate.characterised_occurrence_count > 0 and aggregate.resolved_categories >= selected_category_ids:
        cf_status = "characterised"
    else:
        cf_status = "partly_characterised"
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
    row["cf_status"] = cf_status
    return row


def create_flow_priority(
    config: FlowPriorityConfig,
    *,
    cf_prompt: Optional[PromptCallback] = None,
    progress_callback: Optional[ProgressCallback] = None,
) -> FlowPriorityResult:
    audit_tau_values = _normalise_tau_values(config.audit_tau_values)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "lcia_flow_priority.csv"
    metadata_path = output_dir / "lcia_flow_priority_metadata.json"
    warnings: List[WarningRecord] = []
    cf_ambiguities: List[CFAmbiguityRecord] = []
    selected_categories: List[ImpactCategory] = []
    empty_selected_categories: List[ImpactCategory] = []
    flow_aggregates: Dict[str, _FlowAggregate] = {}
    resolution_manager = CFResolutionManager(
        mode="gui" if cf_prompt is not None else "cli",
        choices_path=config.cf_resolution_file,
        prompt=cf_prompt,
    )
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

    _emit_progress(
        progress_callback,
        step="load_methods",
        message="Step 2/6: Scanning optional methods input...",
        current=current_step,
        total=total_steps,
    )
    methods_archive = (
        index_archive(config.methods, require_processes=False, require_flows=False)
        if config.methods
        else None
    )
    current_step += 1
    _emit_progress(
        progress_callback,
        step="load_methods",
        message=(
            "Step 2/6: Optional methods indexed."
            if methods_archive is not None
            else "Step 2/6: No external methods input provided."
        ),
        current=current_step,
        total=total_steps,
    )

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
    current_step += 1
    _emit_progress(
        progress_callback,
        step="select_categories",
        message=f"Step 4/6: Selected {len(selected_categories)} LCIA categories.",
        current=current_step,
        total=total_steps,
    )

    process_total = len(database_archive.processes)
    process_lookup = {locator.path: locator for locator in database_archive.processes.values()}
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
    selected_category_ids = {category.category_id for category in selected_categories}
    for rel_path, raw_bytes in iter_source_entries(database_archive.resolved_source_path):
        locator = process_lookup.get(rel_path)
        if locator is None:
            continue
        process_data = parse_json_object(raw_bytes, rel_path)
        exchanges = list(process_data.get("exchanges") or [])
        matrix, candidate_indices, exchange_keys, _characterised_flags, resolved_mask = build_contribution_details(
            exchanges=exchanges,
            flow_lookup=database_archive.flows,
            categories=selected_categories,
            unit_registry=database_archive.units,
            strict_units=config.strict_units,
            tol=config.tolerance,
            process_data=process_data,
            warning_records=warnings,
            ambiguity_records=cf_ambiguities,
            diagnostic_file="lcia_flow_priority_metadata.json",
            resolution_manager=resolution_manager,
        )
        flow_ids: List[str] = []
        characterised_mask = resolved_mask.any(axis=0) if resolved_mask.size else np.zeros(len(candidate_indices), dtype=bool)
        nonzero_characterised = (
            np.any(np.abs(matrix) > config.tolerance, axis=0)
            if matrix.size
            else np.zeros(len(candidate_indices), dtype=bool)
        )
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
                unit_registry=database_archive.units,
                audit_tau_values=audit_tau_values,
            )
            aggregate.occurrence_count += 1
            if characterised_mask[local_index]:
                aggregate.characterised_occurrence_count += 1
            for row_index, category in enumerate(selected_categories):
                if resolved_mask.shape[0] > row_index and resolved_mask[row_index, local_index]:
                    aggregate.resolved_categories.add(category.category_id)
            flow_ids.append(flow_id)

        positive_ladder = build_greedy_ladder(np.maximum(matrix, 0.0), exchange_keys=exchange_keys, tol=config.tolerance)
        negative_ladder = build_greedy_ladder(np.maximum(-matrix, 0.0), exchange_keys=exchange_keys, tol=config.tolerance)
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

        _update_flow_metrics_for_sign(
            np.maximum(matrix, 0.0),
            positive_ladder,
            flow_ids,
            selected_categories,
            audit_tau_values,
            flow_aggregates,
            config.tolerance,
            process_name=str(process_data.get("name") or locator.name),
            process_id=str(process_data.get("@id") or process_data.get("id") or locator.object_id),
            sign="+",
        )
        _update_flow_metrics_for_sign(
            np.maximum(-matrix, 0.0),
            negative_ladder,
            flow_ids,
            selected_categories,
            audit_tau_values,
            flow_aggregates,
            config.tolerance,
            process_name=str(process_data.get("name") or locator.name),
            process_id=str(process_data.get("@id") or process_data.get("id") or locator.object_id),
            sign="-",
        )

        process_index += 1
        now = time.monotonic()
        if process_index == 1 or process_index == process_total or now - last_progress_at >= 0.15:
            last_progress_at = now
            _emit_progress(
                progress_callback,
                step="audit_processes",
                message=_process_progress_message(process_index, process_total, str(process_data.get("name") or locator.name)),
                current=current_step,
                total=total_steps,
                process_current=process_index,
                process_total=process_total,
                process_name=str(process_data.get("name") or locator.name),
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
    fieldnames = [*_BASE_COLUMNS, *metric_columns, "cf_status"]
    rows = [
        _csv_row(flow_aggregates[flow_id], audit_tau_values, selected_category_ids)
        for flow_id in sorted(flow_aggregates)
    ]
    write_manifest_csv(csv_path, rows, fieldnames)
    metadata = {
        "file_type": "lcia_flow_priority",
        "schema_version": 1,
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
        "cf_resolution_policy": {
            "mode": "gui" if cf_prompt is not None else "cli",
            "saved_choices_path": config.cf_resolution_file or "",
            "saved_choice_reuse_enabled": bool(config.cf_resolution_file),
            "interactive_prompt_enabled": bool(cf_prompt is not None),
        },
        "cf_resolution_choices_hash": _choice_history_hash(resolution_manager),
        "algorithm": "nested_greedy_lcia_flow_priority",
        "algorithm_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "csv_filename": csv_path.name,
        "csv_columns": fieldnames,
        "n_processes_total": process_total,
        "n_elementary_occurrences_total": sum(item.occurrence_count for item in flow_aggregates.values()),
        "n_flows_ranked": len(rows),
        "n_warning_records": len(warnings),
        "n_cf_ambiguity_records": len(cf_ambiguities),
        "cf_resolution_summary": resolution_manager.summary.__dict__,
        "notes": [
            "eta is the single-flow certificate shortfall after overshoot margin is subtracted.",
            "loss_max is the maximum raw retained-coverage loss caused by omitting one flow.",
            "Group eta cannot be computed exactly by summing single-flow eta values.",
            "Summed loss_max values provide a conservative upper bound for group-risk screening.",
        ],
        "warnings": [warning.__dict__ for warning in warnings],
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
