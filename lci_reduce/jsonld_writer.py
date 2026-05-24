"""JSON-LD output writing and orchestration."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Optional
from zipfile import ZIP_DEFLATED, ZipFile

from . import __version__
from .cf_resolution import CFResolutionManager, PromptCallback
from .errors import DataFormatError
from .jsonld_reader import index_archive, iter_source_entries, parse_json_object, read_source_bytes
from .lcia import (
    collect_categories,
    ensure_category_factor_quality,
    impact_category_report,
    resolve_lcia_archives,
    select_lcia_categories,
)
from .manifest import ManifestWriter
from .models import CFAmbiguityRecord, CreateConfig, CreateProgressUpdate, CreateResult, WarningRecord
from .reducer import reduce_process
from .validation import ReductionRunSummary, build_run_summary


ProgressCallback = Callable[[CreateProgressUpdate], None]


def _selection_slug(selection: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in selection).strip("_").lower() or "selection"


def _output_zip_name(input_name: str, tau: float, selection: str) -> str:
    stem = Path(input_name).stem
    return f"{stem}_lite_tau_{tau}_{_selection_slug(selection)}.zip"


def _partial_zip_path(output_zip: Path) -> Path:
    return output_zip.with_name(f".{output_zip.name}.partial")


def _serialise_json(data: dict) -> bytes:
    return json.dumps(data, indent=2, ensure_ascii=True, sort_keys=False).encode("utf-8")


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
    tau: float | None = None,
    n_elementary_before: int | None = None,
    n_elementary_removed: int | None = None,
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
            tau=tau,
            n_elementary_before=n_elementary_before,
            n_elementary_removed=n_elementary_removed,
        )
    )


def _copy_optional_method_entries(
    output_archive: ZipFile,
    *,
    database_source: str,
    database_paths: set[str],
    methods_source: str,
) -> None:
    for rel_path, raw_bytes in iter_source_entries(methods_source):
        if rel_path in database_paths:
            if read_source_bytes(database_source, rel_path) != raw_bytes:
                raise DataFormatError(f"Conflicting optional methods entry path: {rel_path}")
            continue
        output_archive.writestr(rel_path, raw_bytes)


def _process_progress_message(
    *,
    process_index: int,
    process_total: int,
    process_name: str,
    tau: float,
    summary: ReductionRunSummary,
) -> str:
    removed = summary.n_elementary_before - summary.n_elementary_after
    label = process_name or "-"
    return (
        f"Step 5/8: Process {process_index}/{process_total} | "
        f"tau={tau:.4g} | removed {removed}/{summary.n_elementary_before} elementary exchanges | "
        f"{label}"
    )


def create_lite_database(
    config: CreateConfig,
    *,
    cf_prompt: Optional[PromptCallback] = None,
    progress_callback: Optional[ProgressCallback] = None,
) -> CreateResult:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    exchange_manifest_path = output_dir / "exchange_manifest.csv"
    process_manifest_path = output_dir / "process_manifest.csv"
    run_summary_path = output_dir / "run_summary.json"

    exchange_columns = [
        "database_name",
        "process_id",
        "process_name",
        "exchange_id",
        "exchange_index",
        "flow_id",
        "flow_name",
        "flow_type",
        "compartment",
        "subcompartment",
        "amount",
        "unit",
        "characterised",
        "selected",
        "removed",
        "protected",
        "selected_by_positive_cover",
        "selected_by_negative_cover",
        "uncharacterised",
        "removal_reason",
        "tau",
        "method_selection",
    ]
    process_columns = [
        "process_id",
        "process_name",
        "n_total_exchanges_before",
        "n_total_exchanges_after",
        "n_elementary_before",
        "n_elementary_after",
        "n_elementary_characterised",
        "n_elementary_uncharacterised",
        "n_uncharacterised_kept",
        "n_uncharacterised_removed",
        "n_elementary_removed",
        "n_protected_exchanges",
        "positive_cover_ok",
        "negative_cover_ok",
        "min_positive_coverage",
        "min_negative_coverage",
        "n_active_positive_categories",
        "n_active_negative_categories",
        "coverage_failure",
        "status",
        "warnings",
    ]
    warnings: List[WarningRecord] = []
    cf_ambiguities: List[CFAmbiguityRecord] = []
    selected_categories = []
    empty_selected_categories = []
    database_archive = None
    methods_archive = None
    output_zip = None
    partial_output_zip = None
    summary = ReductionRunSummary()
    lcia_method_source = "database"
    internal_lcia_methods_ignored = False
    resolution_manager = CFResolutionManager(
        mode="gui" if cf_prompt is not None else "cli",
        choices_path=config.cf_resolution_file,
        prompt=cf_prompt,
    )

    try:
        current_step = 0
        total_steps = 8
        _emit_progress(
            progress_callback,
            step="load_database",
            message="Step 1/8: Scanning database archive...",
            current=current_step,
            total=total_steps,
            tau=config.tau,
        )
        database_archive = index_archive(config.database, require_processes=True, require_flows=True)
        current_step += 1
        _emit_progress(
            progress_callback,
            step="load_database",
            message=(
                f"Step 1/8: Indexed {len(database_archive.processes)} processes and "
                f"{len(database_archive.flows)} flows."
            ),
            current=current_step,
            total=total_steps,
            tau=config.tau,
        )

        _emit_progress(
            progress_callback,
            step="load_methods",
            message="Step 2/8: Scanning optional methods input...",
            current=current_step,
            total=total_steps,
            tau=config.tau,
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
                "Step 2/8: Optional methods indexed."
                if methods_archive is not None
                else "Step 2/8: No external methods input provided."
            ),
            current=current_step,
            total=total_steps,
            tau=config.tau,
        )

        output_zip = output_dir / _output_zip_name(database_archive.source_name, config.tau, config.method_selection)
        partial_output_zip = _partial_zip_path(output_zip)
        if partial_output_zip.exists():
            partial_output_zip.unlink()

        archives, lcia_method_source, internal_lcia_methods_ignored = resolve_lcia_archives(
            database_archive,
            methods_archive,
        )
        _emit_progress(
            progress_callback,
            step="collect_categories",
            message="Step 3/8: Collecting LCIA categories and CF candidates...",
            current=current_step,
            total=total_steps,
            tau=config.tau,
        )
        categories = collect_categories(
            archives,
            ambiguity_records=cf_ambiguities,
            diagnostic_file="run_summary.json",
        )
        current_step += 1
        _emit_progress(
            progress_callback,
            step="collect_categories",
            message=f"Step 3/8: Collected {len(categories)} LCIA categories.",
            current=current_step,
            total=total_steps,
            tau=config.tau,
        )

        _emit_progress(
            progress_callback,
            step="select_categories",
            message="Step 4/8: Selecting LCIA categories for this run...",
            current=current_step,
            total=total_steps,
            tau=config.tau,
        )
        selected_categories = select_lcia_categories(categories, config.method_selection)
        empty_selected_categories = ensure_category_factor_quality(selected_categories, warnings)
        current_step += 1
        _emit_progress(
            progress_callback,
            step="select_categories",
            message=f"Step 4/8: Selected {len(selected_categories)} LCIA categories.",
            current=current_step,
            total=total_steps,
            tau=config.tau,
        )

        process_total = len(database_archive.processes)
        process_paths = {locator.path: locator for locator in database_archive.processes.values()}
        database_paths = set(database_archive.entry_paths)
        _emit_progress(
            progress_callback,
            step="reduce_processes",
            message=f"Step 5/8: Reducing and writing {process_total} processes...",
            current=current_step,
            total=total_steps,
            process_current=0,
            process_total=process_total,
            tau=config.tau,
            n_elementary_before=0,
            n_elementary_removed=0,
        )
        with ManifestWriter(exchange_manifest_path, exchange_columns) as exchange_writer:
            with ManifestWriter(process_manifest_path, process_columns) as process_writer:
                with ZipFile(partial_output_zip, "w", compression=ZIP_DEFLATED) as output_archive:
                    last_progress_at = 0.0
                    process_index = 0
                    for rel_path, raw_bytes in iter_source_entries(database_archive.resolved_source_path):
                        process_locator = process_paths.get(rel_path)
                        if process_locator is None:
                            output_archive.writestr(rel_path, raw_bytes)
                            continue

                        process_data = parse_json_object(raw_bytes, rel_path)
                        result = reduce_process(
                            process_data=process_data,
                            flow_lookup=database_archive.flows,
                            categories=selected_categories,
                            unit_registry=database_archive.units,
                            tau=config.tau,
                            uncharacterised_policy=config.uncharacterised_policy,
                            strict_units=config.strict_units,
                            tol=config.tolerance,
                            database_name=database_archive.source_name,
                            warning_records=warnings,
                            ambiguity_records=cf_ambiguities,
                            diagnostic_file="run_summary.json",
                            resolution_manager=resolution_manager,
                        )
                        result.process_path = rel_path
                        for row in result.elementary_rows:
                            row["method_selection"] = config.method_selection
                        exchange_writer.write_rows(result.elementary_rows)
                        process_writer.write_row(result.process_row)
                        output_archive.writestr(rel_path, _serialise_json(result.reduced_process))

                        summary.add_process_result(result)
                        process_index += 1
                        now = time.monotonic()
                        if process_index == 1 or process_index == process_total or now - last_progress_at >= 0.15:
                            last_progress_at = now
                            removed = summary.n_elementary_before - summary.n_elementary_after
                            process_name = result.process_name or result.process_id or process_locator.name
                            _emit_progress(
                                progress_callback,
                                step="reduce_processes",
                                message=_process_progress_message(
                                    process_index=process_index,
                                    process_total=process_total,
                                    process_name=process_name,
                                    tau=config.tau,
                                    summary=summary,
                                ),
                                current=current_step,
                                total=total_steps,
                                process_current=process_index,
                                process_total=process_total,
                                process_name=process_name,
                                tau=config.tau,
                                n_elementary_before=summary.n_elementary_before,
                                n_elementary_removed=removed,
                            )

                    if methods_archive is not None and not internal_lcia_methods_ignored:
                        _copy_optional_method_entries(
                            output_archive,
                            database_source=database_archive.resolved_source_path,
                            database_paths=database_paths,
                            methods_source=methods_archive.resolved_source_path,
                        )
        current_step += 1
        _emit_progress(
            progress_callback,
            step="reduce_processes",
            message=(
                f"Step 5/8: Finished {process_total} processes. "
                f"Removed {summary.n_elementary_before - summary.n_elementary_after}/"
                f"{summary.n_elementary_before} elementary exchanges."
            ),
            current=current_step,
            total=total_steps,
            process_current=process_total,
            process_total=process_total,
            tau=config.tau,
            n_elementary_before=summary.n_elementary_before,
            n_elementary_removed=summary.n_elementary_before - summary.n_elementary_after,
        )

        _emit_progress(
            progress_callback,
            step="write_zip",
            message="Step 6/8: Finalising lite database ZIP...",
            current=current_step,
            total=total_steps,
            tau=config.tau,
        )
        partial_output_zip.replace(output_zip)
        current_step += 1
        _emit_progress(
            progress_callback,
            step="write_zip",
            message="Step 6/8: Lite database ZIP ready.",
            current=current_step,
            total=total_steps,
            tau=config.tau,
        )

        _emit_progress(
            progress_callback,
            step="write_manifests",
            message="Step 7/8: Finalising manifests...",
            current=current_step,
            total=total_steps,
            tau=config.tau,
        )
        resolution_manager.write_choices()
        current_step += 1
        _emit_progress(
            progress_callback,
            step="write_manifests",
            message="Step 7/8: Manifests written.",
            current=current_step,
            total=total_steps,
            tau=config.tau,
        )

        _emit_progress(
            progress_callback,
            step="write_summary",
            message="Step 8/8: Building run summary...",
            current=current_step,
            total=total_steps,
            tau=config.tau,
        )
        config_summary = {
            "status": "completed",
            "input_database_zip": config.database,
            "optional_methods_input": config.methods,
            "tau": config.tau,
            "method_selection": config.method_selection,
            "lcia_method_source": lcia_method_source,
            "internal_lcia_methods_ignored": internal_lcia_methods_ignored,
            "selected_lcia_categories": [impact_category_report(category).__dict__ for category in selected_categories],
            "empty_selected_lcia_categories": [
                impact_category_report(category).__dict__ for category in empty_selected_categories
            ],
            "uncharacterised_policy": config.uncharacterised_policy,
            "strict_units": config.strict_units,
            "tolerance": config.tolerance,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "package_version": __version__,
        }
        run_summary = build_run_summary(
            reduced_results=summary,
            selected_categories=selected_categories,
            empty_selected_categories=empty_selected_categories,
            warnings=warnings,
            cf_ambiguities=cf_ambiguities,
            output_zip=str(output_zip),
            exchange_manifest_csv=str(exchange_manifest_path),
            process_manifest_csv=str(process_manifest_path),
            config=config_summary,
            cf_resolution_choices=resolution_manager.choice_history,
            cf_resolution_summary=resolution_manager.summary,
        )
        run_summary_path.write_text(json.dumps(run_summary, indent=2, ensure_ascii=True), encoding="utf-8")
        current_step += 1
        _emit_progress(
            progress_callback,
            step="write_summary",
            message="Step 8/8: Run completed.",
            current=current_step,
            total=total_steps,
            process_current=process_total,
            process_total=process_total,
            tau=config.tau,
            n_elementary_before=summary.n_elementary_before,
            n_elementary_removed=summary.n_elementary_before - summary.n_elementary_after,
        )

        return CreateResult(
            output_zip=str(output_zip),
            exchange_manifest_csv=str(exchange_manifest_path),
            process_manifest_csv=str(process_manifest_path),
            run_summary_json=str(run_summary_path),
            summary=run_summary,
        )
    except Exception as exc:
        _emit_progress(
            progress_callback,
            step="failed",
            message=f"Run failed: {exc}",
            current=0,
            total=1,
            tau=config.tau,
        )
        if partial_output_zip is not None and partial_output_zip.exists():
            partial_output_zip.unlink()
        resolution_manager.write_choices()
        config_summary = {
            "status": "failed",
            "input_database_zip": config.database,
            "optional_methods_input": config.methods,
            "tau": config.tau,
            "method_selection": config.method_selection,
            "lcia_method_source": lcia_method_source,
            "internal_lcia_methods_ignored": internal_lcia_methods_ignored,
            "selected_lcia_categories": [impact_category_report(category).__dict__ for category in selected_categories],
            "empty_selected_lcia_categories": [
                impact_category_report(category).__dict__ for category in empty_selected_categories
            ],
            "uncharacterised_policy": config.uncharacterised_policy,
            "strict_units": config.strict_units,
            "tolerance": config.tolerance,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "package_version": __version__,
            "error": str(exc),
        }
        run_summary = build_run_summary(
            reduced_results=summary,
            selected_categories=selected_categories,
            empty_selected_categories=empty_selected_categories,
            warnings=warnings,
            cf_ambiguities=cf_ambiguities,
            output_zip=str(output_zip) if output_zip else "",
            exchange_manifest_csv=str(exchange_manifest_path),
            process_manifest_csv=str(process_manifest_path),
            config=config_summary,
            cf_resolution_choices=resolution_manager.choice_history,
            cf_resolution_summary=resolution_manager.summary,
        )
        run_summary_path.write_text(json.dumps(run_summary, indent=2, ensure_ascii=True), encoding="utf-8")
        raise
