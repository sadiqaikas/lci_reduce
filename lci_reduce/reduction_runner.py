"""Reduction output writing and orchestration."""

from __future__ import annotations

import gc
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional
from zipfile import ZIP_DEFLATED, ZipFile

from .cf_resolution import CFResolutionManager
from .contribution import exchange_flow_id
from .ecospold1_reader import iter_ecospold_processes, write_reduced_ecospold_archive
from .errors import DataFormatError
from .archive_reader import index_archive, iter_source_entries, merge_unit_registries, parse_json_object
from .lcia import collect_categories, ensure_category_factor_quality, resolve_lcia_archives, select_lcia_categories
from .models import CFAmbiguityRecord, CFAmbiguityStats, CreateConfig, CreateProgressUpdate, CreateResult, FlowInfo, ProcessReductionResult
from .reducer import reduce_process


ProgressCallback = Callable[[CreateProgressUpdate], None]

_FLUSH_EVERY_PROCESSES = 10
_GC_EVERY_PROCESSES = 25


def _output_zip_name() -> str:
    return "reduced_database.zip"


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


def _flush_output_archive(output_archive: ZipFile) -> None:
    fp = getattr(output_archive, "fp", None)
    if fp is not None and hasattr(fp, "flush"):
        fp.flush()


def _process_identity(process_data: dict | None, fallback_id: str, fallback_name: str) -> tuple[str, str]:
    if process_data is None:
        return fallback_id, fallback_name
    process_id = str(process_data.get("@id") or process_data.get("id") or fallback_id)
    process_name = str(process_data.get("name") or fallback_name)
    return process_id, process_name


def _process_basic_counts(process_data: dict | None, flow_lookup: dict[str, FlowInfo]) -> tuple[int | None, int | None]:
    if process_data is None:
        return None, None
    exchanges = list(process_data.get("exchanges") or [])
    n_elementary = 0
    for exchange in exchanges:
        flow_id = exchange_flow_id(exchange)
        flow = flow_lookup.get(flow_id) if flow_id else None
        if flow is not None and flow.is_elementary:
            n_elementary += 1
    return len(exchanges), n_elementary


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


def _raise_if_no_selected_categories(
    *,
    selected_categories: list,
    methods_input: str | None,
    input_parse_warnings: list[dict[str, str]],
) -> None:
    if selected_categories:
        return
    message = "No usable LCIA categories were available for this run."
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


@dataclass
class _RunAggregate:
    n_processes_ok: int = 0
    n_processes_error: int = 0
    n_total_exchanges_before: int = 0
    n_total_exchanges_after: int = 0
    n_elementary_before: int = 0
    n_elementary_after: int = 0
    n_uncharacterised_total: int = 0
    n_uncharacterised_kept: int = 0
    n_uncharacterised_removed: int = 0
    min_positive_coverage_global: float = 1.0
    min_negative_coverage_global: float = 1.0
    has_positive_coverage: bool = False
    has_negative_coverage: bool = False
    cf_ambiguity_stats: CFAmbiguityStats = field(default_factory=CFAmbiguityStats)

    def add_success(self, result: ProcessReductionResult) -> None:
        row = result.process_row
        self.n_processes_ok += 1
        self.n_total_exchanges_before += int(row["n_total_exchanges_before"])
        self.n_total_exchanges_after += int(row["n_total_exchanges_after"])
        self.n_elementary_before += int(row["n_elementary_before"])
        self.n_elementary_after += int(row["n_elementary_after"])
        self.n_uncharacterised_total += int(row["n_elementary_uncharacterised"])
        self.n_uncharacterised_kept += int(row.get("n_uncharacterised_kept", result.n_uncharacterised_kept))
        self.n_uncharacterised_removed += int(row.get("n_uncharacterised_removed", result.n_uncharacterised_removed))
        n_active_positive_rows = int(row.get("n_active_positive_rows", row.get("n_active_positive_categories", 0)))
        n_active_negative_rows = int(row.get("n_active_negative_rows", row.get("n_active_negative_categories", 0)))
        if n_active_positive_rows > 0:
            self.min_positive_coverage_global = min(
                self.min_positive_coverage_global,
                float(row["min_positive_coverage"]),
            )
            self.has_positive_coverage = True
        if n_active_negative_rows > 0:
            self.min_negative_coverage_global = min(
                self.min_negative_coverage_global,
                float(row["min_negative_coverage"]),
            )
            self.has_negative_coverage = True
        self.cf_ambiguity_stats.merge(result.cf_ambiguity_stats)

    def add_error(
        self,
        *,
        n_total_exchanges_before: int | None,
        n_total_exchanges_after: int | None,
        n_elementary_before: int | None,
        n_elementary_after: int | None,
    ) -> None:
        self.n_processes_error += 1
        if n_total_exchanges_before is not None:
            self.n_total_exchanges_before += int(n_total_exchanges_before)
        if n_total_exchanges_after is not None:
            self.n_total_exchanges_after += int(n_total_exchanges_after)
        if n_elementary_before is not None:
            self.n_elementary_before += int(n_elementary_before)
        if n_elementary_after is not None:
            self.n_elementary_after += int(n_elementary_after)

    @property
    def n_elementary_removed(self) -> int:
        return self.n_elementary_before - self.n_elementary_after

    @property
    def reduction_ratio_elementary(self) -> float:
        if self.n_elementary_before <= 0:
            return 0.0
        return self.n_elementary_removed / self.n_elementary_before

    @property
    def min_positive_output(self) -> float:
        return self.min_positive_coverage_global if self.has_positive_coverage else 1.0

    @property
    def min_negative_output(self) -> float:
        return self.min_negative_coverage_global if self.has_negative_coverage else 1.0


def _process_progress_message(
    *,
    process_index: int,
    process_total: int,
    process_name: str,
    tau: float,
    summary: _RunAggregate,
) -> str:
    removed = summary.n_elementary_removed
    label = process_name or "-"
    errors = f" | errors {summary.n_processes_error}" if summary.n_processes_error else ""
    return (
        f"Step 5/8: Process {process_index}/{process_total} | "
        f"tau={tau:.4g} | removed {removed}/{summary.n_elementary_before} elementary exchanges"
        f"{errors} | {label}"
    )


def _write_debug_line(debug_file, payload: dict) -> None:
    debug_file.write(json.dumps(payload, ensure_ascii=True) + "\n")
    debug_file.flush()


def _success_debug_record(
    *,
    process_index: int,
    result: ProcessReductionResult,
    runtime_seconds: float,
) -> dict:
    row = result.process_row
    stats = result.cf_ambiguity_stats
    return {
        "process_index": process_index,
        "process_id": result.process_id,
        "process_name": result.process_name,
        "status": "ok",
        "runtime_seconds": runtime_seconds,
        "n_total_exchanges_before": int(row["n_total_exchanges_before"]),
        "n_total_exchanges_after": int(row["n_total_exchanges_after"]),
        "n_elementary_before": int(row["n_elementary_before"]),
        "n_elementary_after": int(row["n_elementary_after"]),
        "n_elementary_removed": int(row["n_elementary_removed"]),
        "n_elementary_characterised": int(row["n_elementary_characterised"]),
        "n_elementary_uncharacterised": int(row["n_elementary_uncharacterised"]),
        "n_protected_exchanges": int(row["n_protected_exchanges"]),
        "min_positive_coverage": float(row["min_positive_coverage"]),
        "min_negative_coverage": float(row["min_negative_coverage"]),
        "n_active_positive_rows": int(row.get("n_active_positive_rows", row.get("n_active_positive_categories", 0))),
        "n_active_negative_rows": int(row.get("n_active_negative_rows", row.get("n_active_negative_categories", 0))),
        "cf_candidate_sets": stats.n_finite_cf_candidate_sets,
        "regional_scenario_groups": stats.n_regional_scenario_groups,
        "independent_candidate_groups": stats.n_independent_candidate_groups,
        "scenario_rows_added": stats.n_scenario_rows_added,
        "error_type": "",
        "error_message": "",
    }


def _error_debug_record(
    *,
    process_index: int,
    process_id: str,
    process_name: str,
    runtime_seconds: float,
    n_total_exchanges_before: int | None,
    n_total_exchanges_after: int | None,
    n_elementary_before: int | None,
    n_elementary_after: int | None,
    exc: Exception,
) -> dict:
    n_elementary_removed = None
    if n_elementary_before is not None and n_elementary_after is not None:
        n_elementary_removed = int(n_elementary_before) - int(n_elementary_after)
    return {
        "process_index": process_index,
        "process_id": process_id,
        "process_name": process_name,
        "status": "error",
        "runtime_seconds": runtime_seconds,
        "n_total_exchanges_before": n_total_exchanges_before,
        "n_total_exchanges_after": n_total_exchanges_after,
        "n_elementary_before": n_elementary_before,
        "n_elementary_after": n_elementary_after,
        "n_elementary_removed": n_elementary_removed,
        "n_elementary_characterised": None,
        "n_elementary_uncharacterised": None,
        "n_protected_exchanges": None,
        "min_positive_coverage": None,
        "min_negative_coverage": None,
        "n_active_positive_rows": None,
        "n_active_negative_rows": None,
        "cf_candidate_sets": None,
        "regional_scenario_groups": None,
        "independent_candidate_groups": None,
        "scenario_rows_added": None,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
    }


def _build_run_summary(
    *,
    config: CreateConfig,
    output_zip: Path,
    debug_path: Path,
    process_total: int,
    summary: _RunAggregate,
    runtime_seconds_total: float,
    lcia_method_source: str,
    internal_lcia_methods_ignored: bool,
    n_lcia_categories_used: int,
    n_empty_lcia_categories: int,
    resolution_manager: CFResolutionManager,
    source_format: str = "jsonld",
    input_parse_warnings: list[dict[str, str]] | None = None,
) -> dict:
    cf_summary = resolution_manager.summary
    parse_warnings = input_parse_warnings or []
    return {
        "input_database": config.database,
        "source_format": source_format,
        "output_database": str(output_zip),
        "tau": config.tau,
        "uncharacterised_policy": config.uncharacterised_policy,
        "method_selection": config.method_selection,
        "allow_water_mass_volume_override": config.allow_water_mass_volume_override,
        "cf_resolution_policy": "finite_scenario_rows",
        "lcia_method_source": lcia_method_source,
        "internal_lcia_methods_ignored": internal_lcia_methods_ignored,
        "n_lcia_categories_used": n_lcia_categories_used,
        "n_empty_lcia_categories": n_empty_lcia_categories,
        "n_processes_total": process_total,
        "n_processes_ok": summary.n_processes_ok,
        "n_processes_error": summary.n_processes_error,
        "n_total_exchanges_before": summary.n_total_exchanges_before,
        "n_total_exchanges_after": summary.n_total_exchanges_after,
        "n_elementary_before": summary.n_elementary_before,
        "n_elementary_after": summary.n_elementary_after,
        "n_elementary_removed": summary.n_elementary_removed,
        "reduction_ratio_elementary": summary.reduction_ratio_elementary,
        "n_uncharacterised_total": summary.n_uncharacterised_total,
        "n_uncharacterised_kept": summary.n_uncharacterised_kept,
        "n_uncharacterised_removed": summary.n_uncharacterised_removed,
        "n_cf_ambiguities_found": cf_summary.n_cf_ambiguities_found,
        "n_cf_ambiguity_keys_unique": cf_summary.n_cf_ambiguity_keys_unique,
        "n_cf_ambiguities_resolved_automatically": cf_summary.n_cf_ambiguities_resolved_automatically,
        "n_cf_ambiguities_unresolved": cf_summary.n_cf_ambiguities_unresolved,
        "n_finite_cf_candidate_sets": summary.cf_ambiguity_stats.n_finite_cf_candidate_sets,
        "n_scenario_rows_added": summary.cf_ambiguity_stats.n_scenario_rows_added,
        "n_regional_scenario_groups": summary.cf_ambiguity_stats.n_regional_scenario_groups,
        "n_independent_candidate_groups": summary.cf_ambiguity_stats.n_independent_candidate_groups,
        "n_unresolved_cf_ambiguities": summary.cf_ambiguity_stats.n_unresolved_cf_ambiguities,
        "max_candidate_set_size": summary.cf_ambiguity_stats.max_candidate_set_size,
        "min_positive_coverage_global": summary.min_positive_output,
        "min_negative_coverage_global": summary.min_negative_output,
        "n_input_parse_warnings": len(parse_warnings),
        "input_parse_warnings": parse_warnings,
        "runtime_seconds_total": runtime_seconds_total,
        "output_zip_size_bytes": output_zip.stat().st_size if output_zip.exists() else 0,
        "debug_file_size_bytes": debug_path.stat().st_size if debug_path.exists() else 0,
    }


def create_lite_database(
    config: CreateConfig,
    *,
    progress_callback: Optional[ProgressCallback] = None,
) -> CreateResult:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_zip = output_dir / _output_zip_name()
    partial_output_zip = _partial_zip_path(output_zip)
    debug_path = output_dir / "reduction_debug.ndjson"
    run_summary_path = output_dir / "run_summary.json"

    database_archive = None
    methods_archive = None
    selected_categories = []
    cf_ambiguities: list[CFAmbiguityRecord] = []
    process_total = 0
    summary = _RunAggregate()
    lcia_method_source = "database"
    internal_lcia_methods_ignored = False
    resolution_manager = CFResolutionManager(mode="cli")
    run_started_at = time.monotonic()
    input_parse_warnings: list[dict[str, str]] = []

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
            index_archive(
                config.methods,
                require_processes=False,
                require_flows=False,
                ecospold_xml_error_policy="warn_skip",
            )
            if config.methods
            else None
        )
        input_parse_warnings = _archive_parse_warnings(database_archive)
        input_parse_warnings.extend(_archive_parse_warnings(methods_archive))
        current_step += 1
        methods_warning_suffix = ""
        if methods_archive is not None:
            methods_parse_warnings = _archive_parse_warnings(methods_archive)
            if methods_parse_warnings:
                methods_warning_suffix = (
                    f" Skipped {len(methods_parse_warnings)} malformed EcoSpold1 XML file(s) in the optional methods input."
                )
        _emit_progress(
            progress_callback,
            step="load_methods",
            message=(
                f"Step 2/8: Optional methods indexed.{methods_warning_suffix}"
                if methods_archive is not None
                else "Step 2/8: No external methods input provided."
            ),
            current=current_step,
            total=total_steps,
            tau=config.tau,
        )

        if partial_output_zip.exists():
            partial_output_zip.unlink()
        if debug_path.exists():
            debug_path.unlink()

        archives, lcia_method_source, internal_lcia_methods_ignored = resolve_lcia_archives(
            database_archive,
            methods_archive,
        )
        active_unit_registry = merge_unit_registries(
            getattr(database_archive, "units", {}),
            getattr(methods_archive, "units", {}) if methods_archive is not None else {},
        )
        _emit_progress(
            progress_callback,
            step="collect_categories",
            message="Step 3/8: Collecting LCIA categories and CF candidates...",
            current=current_step,
            total=total_steps,
            tau=config.tau,
        )
        categories = collect_categories(archives, diagnostic_file="run_summary.json")
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
        empty_selected_categories = ensure_category_factor_quality(selected_categories)
        _raise_if_no_selected_categories(
            selected_categories=list(selected_categories),
            methods_input=config.methods,
            input_parse_warnings=input_parse_warnings,
        )
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

        if database_archive.source_format == "ecospold1":
            reduced_results: dict[str, ProcessReductionResult] = {}
            with debug_path.open("w", encoding="utf-8", buffering=1) as debug_file:
                last_progress_at = 0.0
                process_index = 0
                for process_locator, process_data in iter_ecospold_processes(database_archive):
                    process_index += 1
                    process_started_at = time.monotonic()
                    result = None
                    progress_process_name = process_locator.name or process_locator.object_id
                    try:
                        result = reduce_process(
                            process_data=process_data,
                            flow_lookup=database_archive.flows,
                            categories=selected_categories,
                            unit_registry=active_unit_registry,
                            tau=config.tau,
                            uncharacterised_policy=config.uncharacterised_policy,
                            strict_units=config.strict_units,
                            tol=config.tolerance,
                            database_name=database_archive.source_name,
                            allow_water_mass_volume_override=config.allow_water_mass_volume_override,
                            diagnostic_file="run_summary.json",
                            resolution_manager=resolution_manager,
                            ambiguity_records=cf_ambiguities,
                            max_scenario_rows_per_process=config.max_scenario_rows_per_process,
                            max_candidate_set_size=config.max_candidate_set_size,
                            return_payload="minimal",
                        )
                        result.process_path = process_locator.path
                        reduced_results[process_locator.path] = result
                        runtime_seconds = time.monotonic() - process_started_at
                        _write_debug_line(
                            debug_file,
                            _success_debug_record(
                                process_index=process_index,
                                result=result,
                                runtime_seconds=runtime_seconds,
                            ),
                        )
                        summary.add_success(result)
                        progress_process_name = result.process_name or result.process_id or progress_process_name
                    except Exception as exc:
                        runtime_seconds = time.monotonic() - process_started_at
                        fallback_id = process_locator.object_id or ""
                        fallback_name = process_locator.name or ""
                        process_id, process_name = _process_identity(process_data, fallback_id, fallback_name)
                        progress_process_name = process_name or process_id or progress_process_name
                        n_total_before, n_elementary_before = _process_basic_counts(process_data, database_archive.flows)
                        if config.fail_fast:
                            summary.add_error(
                                n_total_exchanges_before=n_total_before,
                                n_total_exchanges_after=None,
                                n_elementary_before=n_elementary_before,
                                n_elementary_after=None,
                            )
                            _write_debug_line(
                                debug_file,
                                _error_debug_record(
                                    process_index=process_index,
                                    process_id=process_id,
                                    process_name=process_name,
                                    runtime_seconds=runtime_seconds,
                                    n_total_exchanges_before=n_total_before,
                                    n_total_exchanges_after=None,
                                    n_elementary_before=n_elementary_before,
                                    n_elementary_after=None,
                                    exc=exc,
                                ),
                            )
                            raise
                        summary.add_error(
                            n_total_exchanges_before=n_total_before,
                            n_total_exchanges_after=n_total_before,
                            n_elementary_before=n_elementary_before,
                            n_elementary_after=n_elementary_before,
                        )
                        _write_debug_line(
                            debug_file,
                            _error_debug_record(
                                process_index=process_index,
                                process_id=process_id,
                                process_name=process_name,
                                runtime_seconds=runtime_seconds,
                                n_total_exchanges_before=n_total_before,
                                n_total_exchanges_after=n_total_before,
                                n_elementary_before=n_elementary_before,
                                n_elementary_after=n_elementary_before,
                                exc=exc,
                            ),
                        )
                    finally:
                        if process_index % _GC_EVERY_PROCESSES == 0:
                            gc.collect()
                        result = None

                    now = time.monotonic()
                    if process_index == 1 or process_index == process_total or now - last_progress_at >= 0.15:
                        last_progress_at = now
                        _emit_progress(
                            progress_callback,
                            step="reduce_processes",
                            message=_process_progress_message(
                                process_index=process_index,
                                process_total=process_total,
                                process_name=progress_process_name,
                                tau=config.tau,
                                summary=summary,
                            ),
                            current=current_step,
                            total=total_steps,
                            process_current=process_index,
                            process_total=process_total,
                            process_name=progress_process_name,
                            tau=config.tau,
                            n_elementary_before=summary.n_elementary_before,
                            n_elementary_removed=summary.n_elementary_removed,
                        )

                if methods_archive is not None and not internal_lcia_methods_ignored:
                    _emit_progress(
                        progress_callback,
                        step="reduce_processes",
                        message="Step 5/8: Finalising reduced database archive...",
                        current=current_step,
                        total=total_steps,
                        process_current=process_total,
                        process_total=process_total,
                        tau=config.tau,
                        n_elementary_before=summary.n_elementary_before,
                        n_elementary_removed=summary.n_elementary_removed,
                    )
                write_reduced_ecospold_archive(
                    config.database,
                    partial_output_zip,
                    reduced_results,
                    process_source_map=database_archive.extra.get("process_source_map", {}),
                )
        else:
            process_paths = {locator.path: locator for locator in database_archive.processes.values()}
            with debug_path.open("w", encoding="utf-8", buffering=1) as debug_file:
                with ZipFile(partial_output_zip, "w", compression=ZIP_DEFLATED) as output_archive:
                    last_progress_at = 0.0
                    process_index = 0
                    for rel_path, raw_bytes in iter_source_entries(database_archive.resolved_source_path):
                        process_locator = process_paths.get(rel_path)
                        if process_locator is None:
                            output_archive.writestr(rel_path, raw_bytes)
                            continue

                        process_index += 1
                        process_started_at = time.monotonic()
                        process_data = None
                        result = None
                        progress_process_name = process_locator.name or process_locator.object_id
                        try:
                            process_data = parse_json_object(raw_bytes, rel_path)
                            result = reduce_process(
                                process_data=process_data,
                                flow_lookup=database_archive.flows,
                                categories=selected_categories,
                                unit_registry=active_unit_registry,
                                tau=config.tau,
                                uncharacterised_policy=config.uncharacterised_policy,
                                strict_units=config.strict_units,
                                tol=config.tolerance,
                                database_name=database_archive.source_name,
                                allow_water_mass_volume_override=config.allow_water_mass_volume_override,
                                diagnostic_file="run_summary.json",
                                resolution_manager=resolution_manager,
                                ambiguity_records=cf_ambiguities,
                                max_scenario_rows_per_process=config.max_scenario_rows_per_process,
                                max_candidate_set_size=config.max_candidate_set_size,
                                return_payload="minimal",
                            )
                            result.process_path = rel_path
                            output_archive.writestr(rel_path, _serialise_json(result.reduced_process))
                            runtime_seconds = time.monotonic() - process_started_at
                            _write_debug_line(
                                debug_file,
                                _success_debug_record(
                                    process_index=process_index,
                                    result=result,
                                    runtime_seconds=runtime_seconds,
                                ),
                            )
                            summary.add_success(result)
                            progress_process_name = result.process_name or result.process_id or progress_process_name
                        except Exception as exc:
                            runtime_seconds = time.monotonic() - process_started_at
                            fallback_id = process_locator.object_id or ""
                            fallback_name = process_locator.name or ""
                            process_id, process_name = _process_identity(process_data, fallback_id, fallback_name)
                            progress_process_name = process_name or process_id or progress_process_name
                            n_total_before, n_elementary_before = _process_basic_counts(process_data, database_archive.flows)
                            if config.fail_fast:
                                summary.add_error(
                                    n_total_exchanges_before=n_total_before,
                                    n_total_exchanges_after=None,
                                    n_elementary_before=n_elementary_before,
                                    n_elementary_after=None,
                                )
                                _write_debug_line(
                                    debug_file,
                                    _error_debug_record(
                                        process_index=process_index,
                                        process_id=process_id,
                                        process_name=process_name,
                                        runtime_seconds=runtime_seconds,
                                        n_total_exchanges_before=n_total_before,
                                        n_total_exchanges_after=None,
                                        n_elementary_before=n_elementary_before,
                                        n_elementary_after=None,
                                        exc=exc,
                                    ),
                                )
                                raise
                            output_archive.writestr(rel_path, raw_bytes)
                            summary.add_error(
                                n_total_exchanges_before=n_total_before,
                                n_total_exchanges_after=n_total_before,
                                n_elementary_before=n_elementary_before,
                                n_elementary_after=n_elementary_before,
                            )
                            _write_debug_line(
                                debug_file,
                                _error_debug_record(
                                    process_index=process_index,
                                    process_id=process_id,
                                    process_name=process_name,
                                    runtime_seconds=runtime_seconds,
                                    n_total_exchanges_before=n_total_before,
                                    n_total_exchanges_after=n_total_before,
                                    n_elementary_before=n_elementary_before,
                                    n_elementary_after=n_elementary_before,
                                    exc=exc,
                                ),
                            )
                        finally:
                            if process_index == process_total or process_index % _FLUSH_EVERY_PROCESSES == 0:
                                _flush_output_archive(output_archive)
                            if process_index % _GC_EVERY_PROCESSES == 0:
                                gc.collect()
                            result = None
                            process_data = None

                        now = time.monotonic()
                        if process_index == 1 or process_index == process_total or now - last_progress_at >= 0.15:
                            last_progress_at = now
                            _emit_progress(
                                progress_callback,
                                step="reduce_processes",
                                message=_process_progress_message(
                                    process_index=process_index,
                                    process_total=process_total,
                                    process_name=progress_process_name,
                                    tau=config.tau,
                                    summary=summary,
                                ),
                                current=current_step,
                                total=total_steps,
                                process_current=process_index,
                                process_total=process_total,
                                process_name=progress_process_name,
                                tau=config.tau,
                                n_elementary_before=summary.n_elementary_before,
                                n_elementary_removed=summary.n_elementary_removed,
                            )

                    if methods_archive is not None and not internal_lcia_methods_ignored:
                        _emit_progress(
                            progress_callback,
                            step="reduce_processes",
                            message="Step 5/8: Finalising reduced database archive...",
                            current=current_step,
                            total=total_steps,
                            process_current=process_total,
                            process_total=process_total,
                            tau=config.tau,
                            n_elementary_before=summary.n_elementary_before,
                            n_elementary_removed=summary.n_elementary_removed,
                        )
                    _flush_output_archive(output_archive)

        current_step += 1
        _emit_progress(
            progress_callback,
            step="reduce_processes",
            message=(
                f"Step 5/8: Finished {process_total} processes. "
                f"Removed {summary.n_elementary_removed}/{summary.n_elementary_before} elementary exchanges."
            ),
            current=current_step,
            total=total_steps,
            process_current=process_total,
            process_total=process_total,
            tau=config.tau,
            n_elementary_before=summary.n_elementary_before,
            n_elementary_removed=summary.n_elementary_removed,
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
            step="write_debug",
            message="Step 7/8: Finalising debug log...",
            current=current_step,
            total=total_steps,
            tau=config.tau,
        )
        current_step += 1
        _emit_progress(
            progress_callback,
            step="write_debug",
            message="Step 7/8: Debug log ready.",
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
        runtime_seconds_total = time.monotonic() - run_started_at
        run_summary = _build_run_summary(
            config=config,
            output_zip=output_zip,
            debug_path=debug_path,
            process_total=process_total,
            summary=summary,
            runtime_seconds_total=runtime_seconds_total,
            lcia_method_source=lcia_method_source,
            internal_lcia_methods_ignored=internal_lcia_methods_ignored,
            n_lcia_categories_used=len(selected_categories),
            n_empty_lcia_categories=len(empty_selected_categories),
            resolution_manager=resolution_manager,
            source_format=database_archive.source_format if database_archive is not None else "jsonld",
            input_parse_warnings=input_parse_warnings,
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
            n_elementary_removed=summary.n_elementary_removed,
        )

        return CreateResult(
            output_zip=str(output_zip),
            run_summary_json=str(run_summary_path),
            reduction_debug_ndjson=str(debug_path),
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
        if partial_output_zip.exists():
            partial_output_zip.unlink()
        runtime_seconds_total = time.monotonic() - run_started_at
        run_summary = _build_run_summary(
            config=config,
            output_zip=output_zip,
            debug_path=debug_path,
            process_total=process_total,
            summary=summary,
            runtime_seconds_total=runtime_seconds_total,
            lcia_method_source=lcia_method_source,
            internal_lcia_methods_ignored=internal_lcia_methods_ignored,
            n_lcia_categories_used=len(selected_categories),
            n_empty_lcia_categories=0,
            resolution_manager=resolution_manager,
            source_format=database_archive.source_format if database_archive is not None else "jsonld",
            input_parse_warnings=input_parse_warnings,
        )
        run_summary_path.write_text(json.dumps(run_summary, indent=2, ensure_ascii=True), encoding="utf-8")
        raise
