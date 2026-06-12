"""Compact priority-file analysis for notebooks, scripts, and tests.

This analyser is intentionally *compact-file only*.  It does not reopen the
original database or reconstruct per-process contribution ledgers.  That design
choice keeps the public API lightweight and easy to use from notebooks, but it
also defines the mathematical scope of the results.

From the compact priority CSV we can recover, for each selected flow:

- exact single-flow ``eta`` for the recorded witness context;
- exact single-flow ``loss_max`` upper information;
- a witness string pointing to one process/category/sign context.

What we cannot recover exactly from the compact CSV alone is the joint effect
of removing a *group* of flows everywhere in the database.  The grouped-flow
results reported here are therefore conservative bounds, not exact grouped
omission results.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

from .errors import AnalysisError
from .models import FlowAnalysisResult, PriorityDataset, PriorityRow, PriorityTauPair
from .utils import fold_text


_ETA_RE = re.compile(r"^eta_(?P<token>[0-9A-Za-z_]+)$")
_ALL_RESULT_TYPES = (
    "eta_bounds",
    "loss_max_flows",
    "witness_processes",
    "top_n_repair",
    "least_n_repair",
    "coverage_summary",
)
_RESULT_TYPE_ALIASES = {
    "selected_flows": "loss_max_flows",
}


def _float_or_none(value: str) -> float | None:
    """Parse optional numeric fields from the compact CSV schema.

    The compact CSV stores some fields as blank when a metric is undefined, for
    example ``tau_entry_min`` for a flow that never enters a greedy ladder.
    Returning ``None`` rather than ``0`` preserves that distinction.
    """
    token = (value or "").strip()
    if not token:
        return None
    return float(token)


def _tau_from_token(token: str) -> float:
    """Convert a column token such as `0_95` back into a float tau value.

    The priority writer uses underscores in column names because ``.`` would
    make attribute-like names awkward in CSV headers.  This helper reverses that
    purely syntactic encoding.
    """
    return float(token.replace("_", "."))


def _load_priority_dataset(priority_file: str | Path, metadata_file: str | Path | None) -> PriorityDataset:
    """Load and validate the compact priority CSV plus optional metadata JSON.

    Validation here is deliberately strict because later grouped analysis
    assumes a consistent schema.  In particular, for each detected ``eta`` column
    the function requires matching ``loss_max`` and witness columns.

    The check ``eta <= loss_max`` is important scientifically: a single-flow
    shortfall cannot exceed that same flow's maximum normalised loss in the
    witness context.
    """
    csv_path = Path(priority_file)
    if not csv_path.exists():
        raise AnalysisError(f"Priority file not found: {csv_path}")
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        if not fieldnames:
            raise AnalysisError("Priority CSV is empty")
        tau_pairs: list[PriorityTauPair] = []
        for fieldname in fieldnames:
            match = _ETA_RE.match(fieldname)
            if not match:
                continue
            token = match.group("token")
            if token.endswith("_witness"):
                continue
            loss_column = f"loss_max_{token}"
            witness_column = f"eta_{token}_witness"
            if loss_column not in fieldnames:
                raise AnalysisError(f"Priority CSV is missing {loss_column} for {fieldname}")
            tau_pairs.append(
                PriorityTauPair(
                    tau=_tau_from_token(token),
                    token=token,
                    eta_column=fieldname,
                    loss_column=loss_column,
                    witness_column=witness_column,
                )
            )
        rows: list[PriorityRow] = []
        for raw in reader:
            cleaned = {key: (raw.get(key) or "").strip() for key in fieldnames}
            metrics: dict[str, tuple[float, float, str]] = {}
            for pair in tau_pairs:
                eta = float(cleaned[pair.eta_column] or 0.0)
                loss = float(cleaned[pair.loss_column] or 0.0)
                if eta > loss + 1e-12:
                    raise AnalysisError(f"{pair.eta_column} greater than {pair.loss_column} for flow {cleaned.get('flow_id', '')}")
                metrics[pair.token] = (eta, loss, cleaned.get(pair.witness_column, ""))
            rows.append(
                PriorityRow(
                    raw=cleaned,
                    flow_id=cleaned["flow_id"],
                    flow_name=cleaned["flow_name"],
                    compartment=cleaned["compartment"],
                    subcompartment=cleaned["subcompartment"],
                    reference_unit=cleaned["reference_unit"],
                    occurrence_count=int(cleaned["occurrence_count"] or 0),
                    characterised_occurrence_count=int(cleaned["characterised_occurrence_count"] or 0),
                    tau_entry_min=_float_or_none(cleaned["tau_entry_min"]),
                    tau_entry_median=_float_or_none(cleaned["tau_entry_median"]),
                    tau_entry_max=_float_or_none(cleaned["tau_entry_max"]),
                    metrics=metrics,
                )
            )
    metadata = {}
    metadata_path = ""
    if metadata_file:
        metadata_path = str(metadata_file)
        metadata = json.loads(Path(metadata_file).read_text(encoding="utf-8"))
    return PriorityDataset(csv_path=str(csv_path), rows=rows, tau_pairs=sorted(tau_pairs, key=lambda item: item.tau), metadata_path=metadata_path, metadata=metadata)


def _normalise_result_types(result_types: list[str] | None) -> list[str]:
    """Validate and normalise requested analysis sections.

    ``analyse_flows`` always returns the request bookkeeping fields such as
    matched and unmatched IDs.  ``result_types`` controls only the heavier
    analytical sections.

    ``None`` or ``["all"]`` means "include every supported section".
    ``selected_flows`` is accepted as a backward-compatible alias for
    ``loss_max_flows``.
    """
    if result_types is None:
        return list(_ALL_RESULT_TYPES)
    cleaned = [str(item).strip() for item in result_types if str(item).strip()]
    if not cleaned or any(item == "all" for item in cleaned):
        return list(_ALL_RESULT_TYPES)
    allowed = set(_ALL_RESULT_TYPES)
    normalised: list[str] = []
    for item in cleaned:
        token = _RESULT_TYPE_ALIASES.get(item, item)
        if token not in allowed:
            raise AnalysisError(
                f"Unsupported result type {item!r}. Supported values are: {', '.join(_ALL_RESULT_TYPES)}"
            )
        if token not in normalised:
            normalised.append(token)
    return normalised


def _resolve_requested_rows(
    dataset: PriorityDataset,
    flow_ids: list[str],
    flow_names: list[str],
    *,
    all_flows: bool,
) -> tuple[list[PriorityRow], list[str], list[str], dict[str, list[str]]]:
    """Resolve which compact-priority rows the caller wants to analyse.

    The analyser supports two mutually exclusive selection modes:

    - explicit flow selection through ``flow_ids`` and/or ``flow_names``;
    - whole-file selection through ``all_flows=True``.

    This explicit rule avoids hidden behaviour such as interpreting a special
    flow name like ``"all"``.  If the caller wants every row, they must ask
    for it directly with ``all_flows=True``.
    """
    if all_flows:
        if flow_ids or flow_names:
            raise AnalysisError("all_flows=True cannot be combined with flow_ids or flow_names")
        return list(dataset.rows), [], [], {}
    if not flow_ids and not flow_names:
        raise AnalysisError("Provide flow_ids or flow_names, or set all_flows=True")
    return _match_rows(dataset, flow_ids, flow_names)


def _tau_pair(dataset: PriorityDataset, tau: float) -> PriorityTauPair:
    """Resolve the requested tau to one detected eta/loss column pair.

    The analyser does not interpolate between audit taus.  It only reports
    values that were explicitly computed into the compact CSV.
    """
    for pair in dataset.tau_pairs:
        if abs(pair.tau - tau) <= 1e-12:
            return pair
    raise AnalysisError(f"Audit tau {tau} is not available in the priority CSV")


def _match_rows(dataset: PriorityDataset, flow_ids: list[str], flow_names: list[str]) -> tuple[list[PriorityRow], list[str], list[str], dict[str, list[str]]]:
    """Match user selections by exact flow ID and case-folded flow name.

    Matching rules:

    - flow IDs must match exactly;
    - flow names are compared using case-folded, whitespace-normalised text;
    - ambiguous names are reported rather than guessed.

    Reporting ambiguity explicitly is important for auditability: a notebook
    should never silently collapse multiple distinct elementary flows just
    because they share the same human-readable label.
    """
    rows_by_id = {row.flow_id: row for row in dataset.rows}
    rows_by_name: dict[str, list[PriorityRow]] = {}
    for row in dataset.rows:
        rows_by_name.setdefault(fold_text(row.flow_name), []).append(row)
    matched: list[PriorityRow] = []
    unmatched_ids: list[str] = []
    unmatched_names: list[str] = []
    ambiguous_names: dict[str, list[str]] = {}
    seen_ids: set[str] = set()
    for flow_id in flow_ids:
        row = rows_by_id.get(flow_id)
        if row is None:
            unmatched_ids.append(flow_id)
            continue
        if row.flow_id not in seen_ids:
            matched.append(row)
            seen_ids.add(row.flow_id)
    for flow_name in flow_names:
        candidates = rows_by_name.get(fold_text(flow_name), [])
        if not candidates:
            unmatched_names.append(flow_name)
            continue
        if len(candidates) > 1:
            ambiguous_names[flow_name] = [candidate.flow_id for candidate in candidates]
            continue
        candidate = candidates[0]
        if candidate.flow_id not in seen_ids:
            matched.append(candidate)
            seen_ids.add(candidate.flow_id)
    return matched, unmatched_ids, unmatched_names, ambiguous_names


def _rank_rows(rows: list[PriorityRow], pair: PriorityTauPair, *, reverse: bool) -> list[PriorityRow]:
    """Rank selected rows deterministically by eta, then loss, then identity.

    The extra identity keys ensure that two flows with numerically identical
    metrics still appear in a stable order across runs.
    """
    return sorted(
        rows,
        key=lambda row: (
            row.metrics[pair.token][0],
            row.metrics[pair.token][1],
            row.flow_name.casefold(),
            row.flow_id,
        ),
        reverse=reverse,
    )


def _group_bounds(rows: list[PriorityRow], pair: PriorityTauPair) -> tuple[dict[str, Any], list[str]]:
    """Compute conservative grouped-flow bounds from compact single-flow data.

    Let ``G`` be the selected group of flows and ``tau`` the requested audit
    threshold.

    From the compact CSV we know each flow's single-flow exact witness shortfall
    ``eta_f`` and witness-wise maximum normalised loss ``loss_max_f``.  The
    analyser reports:

    ``lower_bound = max_f eta_f``

    because removing the whole group cannot be less damaging than removing any
    one member alone in its own witness context.

    It also reports the conservative upper screen:

    ``upper_bound = min(tau, min_g [eta_g + sum_{f != g} loss_max_f])``

    This follows the compact-priority-file rule described in the package
    documentation.  The result is conservative because the compact CSV does not
    contain the full row-by-row joint contribution ledger needed for an exact
    grouped omission calculation.
    """
    warnings: list[str] = []
    if not rows:
        return {
            "tau": pair.tau,
            "selected_count": 0,
            "lower_bound_eta": 0.0,
            "upper_bound_eta": 0.0,
            "sum_loss_max": 0.0,
            "exact_eta": 0.0,
        }, warnings
    etas = [row.metrics[pair.token][0] for row in rows]
    losses = [row.metrics[pair.token][1] for row in rows]
    lower = max(etas)
    sum_loss = sum(losses)
    anchored = [eta + sum(loss for i, loss in enumerate(losses) if i != index) for index, eta in enumerate(etas)]
    upper = min(pair.tau, min(anchored))
    exact_eta = None
    if len(rows) == 1:
        exact_eta = etas[0]
    elif upper == 0.0 and lower == 0.0:
        exact_eta = 0.0
    else:
        warnings.append("Grouped eta is conservative only; exact grouped failure requires a contribution ledger.")
    return {
        "tau": pair.tau,
        "selected_count": len(rows),
        "lower_bound_eta": lower,
        "upper_bound_eta": upper,
        "sum_loss_max": sum_loss,
        "exact_eta": exact_eta,
    }, warnings


def analyse_flows(
    priority_file: str,
    metadata_file: str | None = None,
    flow_ids: list[str] | None = None,
    flow_names: list[str] | None = None,
    all_flows: bool = False,
    result_types: list[str] | None = None,
    tau: float = 0.95,
    top_n: int = 20,
) -> FlowAnalysisResult:
    """Analyse selected flows from a compact priority file.

    The returned object is notebook-friendly: it contains matched flows,
    conservative grouped-flow bounds, witness strings, and repair rankings in
    plain Python data structures.

    The function deliberately does not imitate the legacy GUI analyser.  It is
    designed as a direct package API: inputs are plain Python arguments and the
    output is a structured dataclass that is easy to inspect in tests or
    notebooks.

    Supported ``result_types`` values are:

    - ``"eta_bounds"``
    - ``"loss_max_flows"``
    - ``"witness_processes"``
    - ``"top_n_repair"``
    - ``"least_n_repair"``
    - ``"coverage_summary"``

    Request/match bookkeeping fields are always returned.  ``result_types``
    only filters the analytical sections.

    Flow selection rules:

    - use ``flow_ids`` and/or ``flow_names`` to analyse a subset of rows;
    - use ``all_flows=True`` to analyse the entire compact priority CSV;
    - combining ``all_flows=True`` with explicit IDs or names is an error;
    - leaving all three selectors empty is an error.
    """
    dataset = _load_priority_dataset(priority_file, metadata_file)
    requested_result_types = _normalise_result_types(result_types)
    pair = _tau_pair(dataset, tau)
    resolved_flow_ids = list(flow_ids or [])
    resolved_flow_names = list(flow_names or [])
    matched, unmatched_ids, unmatched_names, ambiguous_names = _resolve_requested_rows(
        dataset,
        resolved_flow_ids,
        resolved_flow_names,
        all_flows=all_flows,
    )
    warnings: list[str] = []
    bounds = None
    if "eta_bounds" in requested_result_types:
        bounds, warnings = _group_bounds(matched, pair)
    flow_rows = [
        {
            "flow_id": row.flow_id,
            "flow_name": row.flow_name,
            "compartment": row.compartment,
            "subcompartment": row.subcompartment,
            "reference_unit": row.reference_unit,
            "eta_tau": row.metrics[pair.token][0],
            "loss_max_tau": row.metrics[pair.token][1],
            "eta_witness": row.metrics[pair.token][2],
            "tau_entry_min": row.tau_entry_min,
            "tau_entry_median": row.tau_entry_median,
            "tau_entry_max": row.tau_entry_max,
        }
        for row in matched
    ] if "loss_max_flows" in requested_result_types else None
    # Witness strings are stored compactly in the CSV.  For convenience we
    # expose them again as structured records without pretending that they are a
    # full contribution ledger.
    witness_rows = [
        {
            "flow_id": row.flow_id,
            "flow_name": row.flow_name,
            "eta_witness": row.metrics[pair.token][2],
            "affected_indicator": row.metrics[pair.token][2].split("|")[1].strip() if row.metrics[pair.token][2] else "",
        }
        for row in matched
        if row.metrics[pair.token][2]
    ] if ("witness_processes" in requested_result_types or "coverage_summary" in requested_result_types) else None
    ranked_high = _rank_rows(matched, pair, reverse=True)[:top_n] if "top_n_repair" in requested_result_types else []
    ranked_low = _rank_rows(matched, pair, reverse=False)[:top_n] if "least_n_repair" in requested_result_types else []
    coverage_summary = None
    if "coverage_summary" in requested_result_types:
        coverage_summary = {
            "matched_count": len(matched),
            "characterised_occurrence_count": sum(row.characterised_occurrence_count for row in matched),
            "occurrence_count": sum(row.occurrence_count for row in matched),
            "affected_indicators": sorted(
                {
                    witness["affected_indicator"]
                    for witness in (witness_rows or [])
                    if witness["affected_indicator"]
                }
            ),
        }
    return FlowAnalysisResult(
        requested_flow_ids=resolved_flow_ids,
        requested_flow_names=resolved_flow_names,
        requested_result_types=requested_result_types,
        tau=tau,
        matched_flow_ids=[row.flow_id for row in matched],
        unmatched_flow_ids=unmatched_ids,
        unmatched_flow_names=unmatched_names,
        ambiguous_names=ambiguous_names,
        loss_max_flows=flow_rows,
        selected_flows=flow_rows,
        eta_bounds=bounds,
        witness_processes=witness_rows if "witness_processes" in requested_result_types else None,
        top_n_repair=[
            {"flow_id": row.flow_id, "flow_name": row.flow_name, "eta_tau": row.metrics[pair.token][0], "loss_max_tau": row.metrics[pair.token][1]}
            for row in ranked_high
        ] if "top_n_repair" in requested_result_types else None,
        least_n_repair=[
            {"flow_id": row.flow_id, "flow_name": row.flow_name, "eta_tau": row.metrics[pair.token][0], "loss_max_tau": row.metrics[pair.token][1]}
            for row in ranked_low
        ] if "least_n_repair" in requested_result_types else None,
        coverage_summary=coverage_summary,
        warnings=warnings,
    )
