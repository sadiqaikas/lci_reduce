"""Public reducer workflow for LCIA-conditioned database reduction.

The reducer is intentionally narrow in scope.  It does not rebuild the
database, regenerate UUIDs, or rewrite non-process content.  Its scientific
promise is only this:

- construct characterised contribution rows for the selected LCIA categories;
- run deterministic sign-split greedy tau-cover on each process;
- remove only non-selected elementary exchanges from process exchange lists;
- verify the tau certificate after protection rules are applied.

This module is therefore the package's "action" layer.  It orchestrates
parsing, contribution construction, selection, verification, and file writing,
while delegating the mathematics to :mod:`lca_flowkit.scenarios` and
:mod:`lca_flowkit.cover`.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from .cover import signed_tau_cover, verify_signed_coverage
from .inputs import load_bundle, resolve_categories
from .models import ReductionResult, WarningRecord, ensure_directory
from .reporting import warnings_to_rows, write_csv, write_json
from .scenarios import build_process_contribution_bundle
from .utils import emit_progress
from .writers import write_ecospold1_reduced_zip, write_jsonld_reduced_zip


def _parse_warning_records(bundle_warnings: list[dict[str, str]]) -> list[WarningRecord]:
    """Convert raw parse-warning payloads into typed warning records.

    The readers store lightweight dictionaries in ``bundle.extra`` so they stay
    format-neutral.  The public workflows convert those dictionaries into
    explicit ``WarningRecord`` objects before writing sidecars or returning
    results.
    """
    records: list[WarningRecord] = []
    for item in bundle_warnings:
        records.append(
            WarningRecord(
                code="input_parse_warning",
                message=str(item.get("message") or ""),
                extra={"source_file": str(item.get("source_file") or "")},
            )
        )
    return records


def _reduced_jsonld_process(process: Any, kept_indices: set[int]) -> dict:
    """Return a copied JSON-LD process with only the retained exchanges left.

    ``deepcopy`` is used deliberately.  The reducer must leave the parsed input
    structure untouched so later debugging or comparison code can still inspect
    the original process payload.
    """
    reduced = deepcopy(process)
    exchanges = list(reduced.get("exchanges", []) or [])
    reduced["exchanges"] = [exchange for index, exchange in enumerate(exchanges) if index in kept_indices]
    return reduced


def _selected_categories_payload(categories: list[Any]) -> list[dict[str, str]]:
    """Serialise selected category identity into run metadata.

    Run metadata should show exactly which LCIA scope justified the reduction.
    The reduced archive is *not* certified for methods outside this list.
    """
    return [
        {
            "method_id": category.method_id,
            "method_name": category.method_name,
            "category_id": category.category_id,
            "category_name": category.name,
        }
        for category in categories
    ]


def reduce_database(
    database_path: str,
    methods_path: str | None = None,
    output_dir: str | None = None,
    method_selection: str = "all",
    tau: float = 0.95,
    strict_units: bool = True,
    progress=None,
) -> ReductionResult:
    """Reduce one database archive and write the auditable sidecar outputs.

    Only elementary exchanges are eligible for removal. All other database
    objects are preserved byte-for-byte except for the process exchange lists
    that must be filtered in the output archive.

    The selection rule is:

    ``selected = selected_pos OR selected_neg OR protected``

    where ``selected_pos`` and ``selected_neg`` are obtained from sign-split
    greedy tau-cover and ``protected`` keeps exchanges that policy forbids the
    reducer to drop.

    Regional CF ambiguity is always expanded explicitly into coverage rows.  If
    that expansion grows beyond the configured limits in the scenario builder,
    the workflow raises ``ScenarioExpansionError`` rather than choosing one
    regional factor implicitly.
    """
    output_root = ensure_directory(output_dir or "lca_flowkit_out")
    emit_progress(progress, step="load", message="Loading database input", current=0, total=4)
    bundle = load_bundle(database_path)
    categories = resolve_categories(bundle, methods_path, method_selection)
    warnings: list[WarningRecord] = _parse_warning_records(bundle.extra.get("parse_warnings", []))

    process_updates: dict[str, dict] = {}
    process_keep_indices: dict[str, set[int]] = {}
    process_manifest_rows: list[dict[str, Any]] = []
    exchange_manifest_rows: list[dict[str, Any]] = []
    total_removed = 0

    total = max(len(bundle.processes), 1)
    # Reduction is process-local: each process gets its own characterised
    # contribution bundle and tau certificate.
    for process_index, process in enumerate(bundle.processes, start=1):
        emit_progress(
            progress,
            step="reduce",
            message=f"Reducing process {process_index}/{total}: {process.name}",
            current=process_index,
            total=total,
        )
        contribution_bundle, process_warnings = build_process_contribution_bundle(
            process,
            bundle.flows,
            categories,
            bundle.units,
            strict_units=strict_units,
        )
        warnings.extend(process_warnings)
        elementary_exchanges = [
            exchange
            for exchange in process.exchanges
            if exchange.flow_id in bundle.flows and bundle.flows[exchange.flow_id].is_elementary
        ]
        # Elementary exchanges that never became candidate columns are kept
        # automatically.  That includes, for example, flows that were not found
        # in the parsed flow registry or flows excluded from the characterised
        # matrix construction path.
        keep_elementary = {exchange.index for exchange in elementary_exchanges if exchange.exchange_id not in contribution_bundle.exchange_keys}
        final_selected = np.asarray(contribution_bundle.protected, dtype=bool)
        verification = {
            "positive_cover_ok": True,
            "negative_cover_ok": True,
            "min_positive_coverage": 1.0,
            "min_negative_coverage": 1.0,
            "n_active_positive_rows": 0,
            "n_active_negative_rows": 0,
        }
        if contribution_bundle.exchange_keys:
            result = signed_tau_cover(contribution_bundle.matrix, tau, contribution_bundle.exchange_keys)
            final_selected = result["selected"] | final_selected
            verification = verify_signed_coverage(contribution_bundle.matrix, final_selected, tau)
        # Non-elementary exchanges are always kept.  The reducer's scope is only
        # the elementary subset.
        keep_indices = {exchange.index for exchange in process.exchanges if exchange.flow_id not in bundle.flows or not bundle.flows[exchange.flow_id].is_elementary}
        keep_indices.update(keep_elementary)
        for column_index, exchange_index in enumerate(contribution_bundle.candidate_indices):
            if final_selected[column_index]:
                keep_indices.add(exchange_index)
        removed_here = 0
        column_lookup = {exchange_index: column_index for column_index, exchange_index in enumerate(contribution_bundle.candidate_indices)}
        for exchange in elementary_exchanges:
            column_index = column_lookup.get(exchange.index)
            characterised = bool(column_index is not None and contribution_bundle.characterised[column_index])
            protected = bool(column_index is not None and contribution_bundle.protected[column_index])
            selected = bool(column_index is not None and final_selected[column_index])
            removed = exchange.index not in keep_indices
            if removed:
                removed_here += 1
            exchange_manifest_rows.append(
                {
                    "process_id": process.process_id,
                    "process_name": process.name,
                    "exchange_id": exchange.exchange_id,
                    "flow_id": exchange.flow_id,
                    "flow_name": exchange.flow_name,
                    "characterised": characterised,
                    "protected": protected,
                    "selected": selected,
                    "removed": removed,
                    # ``selected_or_uncharacterised`` covers both selected
                    # characterised exchanges and elementary exchanges that stay
                    # because the chosen LCIA scope did not characterise them.
                    "removal_reason": "greedy_tau_cover" if removed else ("protected" if protected else "selected_or_uncharacterised"),
                }
            )
        total_removed += removed_here
        process_manifest_rows.append(
            {
                "process_id": process.process_id,
                "process_name": process.name,
                "n_exchanges_total": len(process.exchanges),
                "n_elementary_total": len(elementary_exchanges),
                "n_elementary_removed": removed_here,
                "n_elementary_kept": len(elementary_exchanges) - removed_here,
                "positive_cover_ok": verification["positive_cover_ok"],
                "negative_cover_ok": verification["negative_cover_ok"],
                "min_positive_coverage": verification["min_positive_coverage"],
                "min_negative_coverage": verification["min_negative_coverage"],
                "n_active_positive_rows": verification["n_active_positive_rows"],
                "n_active_negative_rows": verification["n_active_negative_rows"],
            }
        )
        if bundle.source_format == "jsonld":
            process_updates[process.source_path] = _reduced_jsonld_process(process.raw, keep_indices)
        else:
            process_keep_indices[process.source_path] = keep_indices

    emit_progress(progress, step="write", message="Writing reduced archive", current=3, total=4)
    output_zip = output_root / ("reduced_database.zip" if bundle.source_format == "jsonld" else "reduced_ecospold1.zip")
    if bundle.source_format == "jsonld":
        written_zip = write_jsonld_reduced_zip(bundle, output_zip, process_updates)
    else:
        written_zip = write_ecospold1_reduced_zip(bundle, output_zip, process_keep_indices)

    metadata = {
        "database_path": database_path,
        "methods_path": methods_path or "",
        "source_format": bundle.source_format,
        "tau": tau,
        "strict_units": strict_units,
        "method_selection": method_selection,
        "ambiguity_policy": {
            "finite_cf": "explicit_scenario_rows",
            "regional_cf": "explicit_scenario_rows",
        },
        "selected_categories": _selected_categories_payload(categories),
        "counts": {
            "n_processes_total": len(bundle.processes),
            "n_elementary_removed": total_removed,
            "n_warnings": len(warnings),
        },
        "assumptions": [
            "Only elementary exchanges were eligible for removal.",
            "Technosphere, product, waste, provider-linked, and quantitative-reference exchanges were preserved.",
            "Single-flow eta values are exact only for single-flow omission contexts.",
            "Grouped-flow risk cannot be inferred exactly from compact single-flow eta values.",
        ],
        "warnings": warnings_to_rows(warnings),
    }
    metadata_json = write_json(output_root / "reduction_metadata.json", metadata)
    warnings_csv = write_csv(
        output_root / "warnings.csv",
        ["code", "message", "process_id", "process_name", "flow_id", "flow_name", "category_id", "category_name", "extra"],
        warnings_to_rows(warnings),
    )
    process_manifest_csv = write_csv(output_root / "process_manifest.csv", list(process_manifest_rows[0].keys()) if process_manifest_rows else ["process_id"], process_manifest_rows)
    exchange_manifest_csv = write_csv(output_root / "exchange_manifest.csv", list(exchange_manifest_rows[0].keys()) if exchange_manifest_rows else ["exchange_id"], exchange_manifest_rows)
    validation_json = write_json(
        output_root / "validation.json",
        {
            "tau": tau,
            "all_processes_passed": all(row["positive_cover_ok"] and row["negative_cover_ok"] for row in process_manifest_rows),
            "processes": process_manifest_rows,
        },
    )
    emit_progress(progress, step="done", message="Reduction complete", current=4, total=4)
    return ReductionResult(
        output_zip=written_zip,
        output_dir=str(output_root),
        metadata_json=metadata_json,
        warnings_csv=warnings_csv,
        process_manifest_csv=process_manifest_csv,
        exchange_manifest_csv=exchange_manifest_csv,
        validation_json=validation_json,
        counts=metadata["counts"],
        warnings=warnings,
        diagnostics={"scenario_rows": sum(row["n_active_positive_rows"] + row["n_active_negative_rows"] for row in process_manifest_rows)},
    )
