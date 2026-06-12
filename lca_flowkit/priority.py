"""Public priority-file workflow for compact LCIA-critical flow auditing.

The priority workflow is a sidecar analysis.  It does not rewrite the source
database and it does not remove exchanges.  Its job is to answer a narrower
question:

``Which elementary flows look LCIA-critical under the same characterised rows
used by reduction?``

The output CSV is intentionally compact: one row per elementary flow, with
single-flow metrics aggregated over all of that flow's occurrences across all
processes.  Because of that compactness, single-flow omission metrics can be
exact in their witness contexts, while grouped-flow analysis later needs
conservative bounds.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .cover import build_greedy_ladder, prefix_length_for_tau
from .inputs import load_bundle, resolve_categories
from .models import PriorityResult, ScenarioRow, WarningRecord, ensure_directory
from .reporting import warnings_to_rows, write_csv, write_json
from .scenarios import build_process_contribution_bundle
from .utils import emit_progress, unique_preserve_order


@dataclass
class _FlowAggregate:
    """Running aggregate for one elementary flow across all processes.

    The public CSV stores one row per flow, not one row per occurrence.  This
    dataclass accumulates the per-occurrence evidence needed to compute that
    compact summary without reopening earlier processes.
    """
    flow_id: str
    flow_name: str
    compartment: str
    subcompartment: str
    reference_unit: str
    occurrence_count: int = 0
    characterised_occurrence_count: int = 0
    tau_entries: list[float] = field(default_factory=list)
    eta_by_tau: dict[float, float] = field(default_factory=dict)
    loss_by_tau: dict[float, float] = field(default_factory=dict)
    witness_by_tau: dict[float, str] = field(default_factory=dict)


def _tau_token(value: float) -> str:
    """Convert a tau value into the compact CSV column suffix form.

    Example:
    ``0.95 -> "0_95"``
    """
    return format(float(value), ".15g").replace(".", "_")


def _single_flow_shortfall(tau: float, coverage_before_loss: float, loss: float) -> float:
    """Exact single-flow shortfall formula for one witness context.

    For one selected flow occurrence ``e`` in one fixed greedy witness context:

    ``eta = max(tau - (coverage_before_loss - loss), 0)``

    Here:

    - ``coverage_before_loss`` is the row coverage with the selected prefix;
    - ``loss`` is that flow occurrence's normalised contribution in the same
      row.

    The formula is exact for removing that one selected flow occurrence from
    that one witness context.
    """
    return max(float(tau) - (float(coverage_before_loss) - float(loss)), 0.0)


def _eta_witness(process_name: str, process_id: str, row: ScenarioRow, sign: str) -> str:
    """Create the compact witness string stored in the priority CSV.

    The CSV keeps witnesses as a compact human-readable string rather than a
    nested JSON object so the file remains spreadsheet-friendly.  The metadata
    still preserves enough structure for a researcher to see which process,
    indicator/scenario, and sign produced the reported maximum shortfall.
    """
    if row.scenario_type == "exact":
        return f"{process_name} | {row.category_name} | {sign}"
    return f"{process_name} | {row.category_name} | {row.scenario_label} | {sign}"


def _update_metrics_for_sign(
    matrix: np.ndarray,
    ladder: Any,
    flow_ids: list[str],
    row_metadata: list[ScenarioRow],
    audit_tau: list[float],
    aggregates: dict[str, _FlowAggregate],
    *,
    process_name: str,
    process_id: str,
    sign: str,
) -> None:
    """Update ``eta`` and ``loss_max`` metrics for one sign-split greedy ladder.

    ``matrix`` is already sign-split and therefore non-negative.  ``ladder``
    records the order in which columns enter the greedy retained prefix.

    For each audit tau:

    1. determine the retained prefix length;
    2. compute current row coverages under that prefix;
    3. for every selected column in the prefix, compute:

       ``loss(row) = contribution(row, e) / full(row)``

       ``eta(row, tau) = max(tau - (coverage(row) - loss(row)), 0)``

    4. retain the maximum ``loss`` and maximum ``eta`` over all active rows as
       the compact single-flow summary for that flow occurrence.

    The outer workflow later aggregates these occurrence-level maxima across all
    processes into one row per flow.
    """
    if not ladder.order or not ladder.active.any():
        return
    retained = np.zeros(matrix.shape[0], dtype=float)
    selected_columns: list[int] = []
    cursor = 0
    active_indices = np.flatnonzero(ladder.active)
    for tau in audit_tau:
        prefix = prefix_length_for_tau(ladder, tau)
        while cursor < prefix:
            local_index = ladder.order[cursor]
            retained += matrix[:, local_index]
            selected_columns.append(local_index)
            cursor += 1
        if not active_indices.size:
            continue
        coverage = retained[active_indices] / ladder.full[active_indices]
        for local_index in selected_columns:
            flow_id = flow_ids[local_index]
            vector = matrix[:, local_index]
            # ``losses`` is the row-wise normalised contribution of this flow
            # occurrence.  It is an upper envelope: removing the occurrence
            # cannot damage a row more than the share it contributes to that
            # row's full magnitude.
            losses = vector[active_indices] / ladder.full[active_indices]
            loss_max = float(losses.max(initial=0.0))
            if loss_max > aggregates[flow_id].loss_by_tau[tau]:
                aggregates[flow_id].loss_by_tau[tau] = loss_max
            shortfalls = np.maximum(tau - (coverage - losses), 0.0)
            eta = float(shortfalls.max(initial=0.0))
            if eta > aggregates[flow_id].eta_by_tau[tau]:
                aggregates[flow_id].eta_by_tau[tau] = eta
                witness_row = int(active_indices[np.argmax(shortfalls)])
                aggregates[flow_id].witness_by_tau[tau] = _eta_witness(process_name, process_id, row_metadata[witness_row], sign)


def _combined_entry_thresholds(positive: np.ndarray, negative: np.ndarray, characterised: list[bool]) -> list[float | None]:
    """Combine positive and negative entry thresholds into one per-flow value.

    One elementary exchange can matter on the positive ladder, the negative
    ladder, both, or neither.  The compact CSV stores one flow-level
    ``tau_entry`` summary, so we use the earliest finite entry threshold across
    the two sign ladders.

    This means ``tau_entry_*`` answers:

    "At what tau does this exchange start becoming necessary on at least one
    sign-split ladder?"
    """
    values: list[float | None] = []
    for index, is_characterised in enumerate(characterised):
        if not is_characterised:
            values.append(None)
            continue
        finite = [value for value in (positive[index], negative[index]) if np.isfinite(value)]
        values.append(float(min(finite)) if finite else None)
    return values


def create_priority_file(
    database_path: str,
    methods_path: str | None = None,
    output_dir: str | None = None,
    method_selection: str = "all",
    audit_tau: list[float] | tuple[float, ...] = (0.95, 0.99),
    strict_units: bool = True,
    progress=None,
) -> PriorityResult:
    """Audit a database and emit a compact LCIA-critical flow priority file.

    The result is a sidecar-only workflow: it does not rewrite the source
    database and it does not attempt grouped-flow exact failure analysis.

    The workflow reuses the same contribution-bundle construction as the
    reducer.  That is deliberate: priority values are only scientifically
    meaningful if they are derived from the same LCIA rows that would govern
    reduction.

    Regional CF ambiguity is always represented explicitly in those coverage
    rows.  If scenario construction becomes too large, the workflow raises
    ``ScenarioExpansionError`` instead of choosing one regional factor.
    """
    tau_values = sorted({float(value) for value in audit_tau})
    if not tau_values or any(value <= 0 or value > 1 for value in tau_values):
        raise ValueError("audit_tau values must be unique and in (0, 1]")
    output_root = ensure_directory(output_dir or "lca_flowkit_priority")
    emit_progress(progress, step="load", message="Loading database input", current=0, total=4)
    bundle = load_bundle(database_path)
    categories = resolve_categories(bundle, methods_path, method_selection)
    warnings = [
        WarningRecord(code="input_parse_warning", message=item.get("message", ""), extra={"source_file": item.get("source_file", "")})
        for item in bundle.extra.get("parse_warnings", [])
    ]
    aggregates: dict[str, _FlowAggregate] = {}

    total = max(len(bundle.processes), 1)
    # Each process contributes local witness contexts that are then aggregated
    # to one row per elementary flow in the compact CSV.
    for process_index, process in enumerate(bundle.processes, start=1):
        emit_progress(
            progress,
            step="priority",
            message=f"Auditing process {process_index}/{total}: {process.name}",
            current=process_index,
            total=total,
        )
        for exchange in process.exchanges:
            flow = bundle.flows.get(exchange.flow_id)
            if flow is None or not flow.is_elementary:
                continue
            aggregate = aggregates.setdefault(
                flow.flow_id,
                _FlowAggregate(
                    flow_id=flow.flow_id,
                    flow_name=flow.name,
                    compartment=flow.compartment,
                    subcompartment=flow.subcompartment,
                    reference_unit=exchange.unit_name,
                    eta_by_tau={tau: 0.0 for tau in tau_values},
                    loss_by_tau={tau: 0.0 for tau in tau_values},
                    witness_by_tau={tau: "" for tau in tau_values},
                ),
            )
            aggregate.occurrence_count += 1

        contribution_bundle, process_warnings = build_process_contribution_bundle(
            process,
            bundle.flows,
            categories,
            bundle.units,
            strict_units=strict_units,
        )
        warnings.extend(process_warnings)
        for local_index, flow_id in enumerate(contribution_bundle.flow_ids):
            if contribution_bundle.characterised[local_index]:
                aggregates[flow_id].characterised_occurrence_count += 1
        if not contribution_bundle.exchange_keys:
            continue
        # Positive and negative ladders are built separately so the priority
        # metrics respect the same sign-split certificate as the reducer.
        positive = np.maximum(contribution_bundle.matrix, 0.0)
        negative = np.maximum(-contribution_bundle.matrix, 0.0)
        positive_ladder = build_greedy_ladder(positive, contribution_bundle.exchange_keys)
        negative_ladder = build_greedy_ladder(negative, contribution_bundle.exchange_keys)
        for local_index, threshold in enumerate(
            _combined_entry_thresholds(
                positive_ladder.entry_thresholds,
                negative_ladder.entry_thresholds,
                contribution_bundle.characterised,
            )
        ):
            if threshold is None:
                continue
            aggregates[contribution_bundle.flow_ids[local_index]].tau_entries.append(threshold)
        _update_metrics_for_sign(
            positive,
            positive_ladder,
            contribution_bundle.flow_ids,
            contribution_bundle.row_metadata,
            tau_values,
            aggregates,
            process_name=process.name,
            process_id=process.process_id,
            sign="pos",
        )
        _update_metrics_for_sign(
            negative,
            negative_ladder,
            contribution_bundle.flow_ids,
            contribution_bundle.row_metadata,
            tau_values,
            aggregates,
            process_name=process.name,
            process_id=process.process_id,
            sign="neg",
        )

    rows: list[dict[str, Any]] = []
    for flow_id in sorted(aggregates):
        aggregate = aggregates[flow_id]
        row = {
            "flow_id": aggregate.flow_id,
            "flow_name": aggregate.flow_name,
            "compartment": aggregate.compartment,
            "subcompartment": aggregate.subcompartment,
            "reference_unit": aggregate.reference_unit,
            "occurrence_count": aggregate.occurrence_count,
            "characterised_occurrence_count": aggregate.characterised_occurrence_count,
            "tau_entry_min": min(aggregate.tau_entries) if aggregate.tau_entries else "",
            "tau_entry_median": statistics.median(aggregate.tau_entries) if aggregate.tau_entries else "",
            "tau_entry_max": max(aggregate.tau_entries) if aggregate.tau_entries else "",
        }
        for tau in tau_values:
            suffix = _tau_token(tau)
            row[f"eta_{suffix}"] = aggregate.eta_by_tau[tau]
            row[f"eta_{suffix}_witness"] = aggregate.witness_by_tau[tau]
            row[f"loss_max_{suffix}"] = aggregate.loss_by_tau[tau]
        rows.append(row)

    priority_csv = write_csv(
        output_root / "lcia_flow_priority.csv",
        list(rows[0].keys()) if rows else [
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
        ],
        rows,
    )
    metadata = {
        "database_path": database_path,
        "methods_path": methods_path or "",
        "source_format": bundle.source_format,
        "strict_units": strict_units,
        "method_selection": method_selection,
        "audit_tau": tau_values,
        "ambiguity_policy": {
            "finite_cf": "explicit_scenario_rows",
            "regional_cf": "explicit_scenario_rows",
        },
        "selected_categories": [
            {
                "method_id": category.method_id,
                "method_name": category.method_name,
                "category_id": category.category_id,
                "category_name": category.name,
            }
            for category in categories
        ],
        "warnings": warnings_to_rows(warnings),
        "counts": {
            "n_processes_total": len(bundle.processes),
            "n_elementary_flows": len(rows),
            "n_warnings": len(warnings),
        },
        "assumptions": [
            "Single-flow eta values are exact only for single-flow omission in the recorded witness context.",
            "Grouped-flow eta cannot be computed exactly from the compact CSV alone.",
            "Use grouped lower and upper bounds for repair prioritisation unless a full contribution ledger is available.",
        ],
    }
    metadata_json = write_json(output_root / "lcia_flow_priority_metadata.json", metadata)
    warnings_csv = write_csv(
        output_root / "warnings.csv",
        ["code", "message", "process_id", "process_name", "flow_id", "flow_name", "category_id", "category_name", "extra"],
        warnings_to_rows(warnings),
    )
    emit_progress(progress, step="done", message="Priority file complete", current=4, total=4)
    return PriorityResult(
        output_dir=str(output_root),
        priority_csv=priority_csv,
        metadata_json=metadata_json,
        warnings_csv=warnings_csv,
        counts=metadata["counts"],
        warnings=warnings,
        diagnostics={"tau_values": tau_values, "flow_ids": unique_preserve_order(row["flow_id"] for row in rows)},
    )
