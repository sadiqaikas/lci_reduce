"""Dedicated LCIA ambiguity exploration workflow."""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Callable, Iterator, List, Optional, Sequence

from .archive_reader import index_archive, iter_source_entries, merge_unit_registries, parse_json_object
from .cf_resolution import CFResolutionManager
from .contribution import build_sparse_contribution_details
from .ecospold1_reader import iter_ecospold_processes
from .errors import LciReduceError
from .lcia import collect_categories, ensure_category_factor_quality, impact_category_report, resolve_lcia_archives, select_lcia_categories
from .models import (
    AmbiguityExploreConfig,
    AmbiguityExploreResult,
    CFAmbiguityRecord,
    CFAmbiguityStats,
    CreateProgressUpdate,
    ImpactCategory,
    WarningRecord,
)
from .validation import (
    _CF_AMBIGUITY_GROUP_CSV_FIELDS,
    _parse_candidate_metadata_json,
    write_cf_ambiguity_group_csv,
)


_DEFAULT_AMBIGUITY_GROUP_CSV_ROW_LIMIT = 10000


ProgressCallback = Callable[[CreateProgressUpdate], None]


class _SpoolingAmbiguitySink:
    def __init__(self) -> None:
        self._temp = NamedTemporaryFile("w+", encoding="utf-8", suffix=".jsonl", delete=False)
        self.path = Path(self._temp.name)
        self.count = 0

    def append(self, record: CFAmbiguityRecord) -> None:
        self._temp.write(json.dumps(asdict(record), ensure_ascii=True, sort_keys=True))
        self._temp.write("\n")
        self.count += 1

    def iter_records(self) -> Iterator[CFAmbiguityRecord]:
        self._temp.flush()
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                payload = json.loads(line)
                yield CFAmbiguityRecord(**payload)

    def close(self) -> None:
        try:
            self._temp.close()
        finally:
            try:
                self.path.unlink(missing_ok=True)
            except Exception:
                pass


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
            process_current=process_current,
            process_total=process_total,
            process_name=process_name,
        )
    )


def _parse_warning_record(source_name: str, source_file: str, message: str, *, input_role: str) -> WarningRecord:
    return WarningRecord(
        severity="warning",
        object_type="database",
        object_id=source_name,
        object_name=source_name,
        process_id="",
        process_name="",
        flow_id="",
        flow_name="",
        category_id="",
        category_name="",
        message=f"{input_role}: {message}",
        source_file=source_file,
    )


def _hash_path(path: str) -> str:
    if not path:
        return ""
    file_path = Path(path)
    hasher = hashlib.sha256()
    if file_path.is_dir():
        for item in sorted(item for item in file_path.rglob("*") if item.is_file()):
            hasher.update(item.relative_to(file_path).as_posix().encode("utf-8"))
            hasher.update(b"\0")
            hasher.update(item.read_bytes())
        return hasher.hexdigest()
    return hashlib.sha256(file_path.read_bytes()).hexdigest()


def _selected_methods(categories: Sequence[ImpactCategory]) -> list[dict[str, str]]:
    seen: dict[tuple[str, str, str, str], dict[str, str]] = {}
    for category in categories:
        key = (
            category.method_id or "",
            category.method_name or "",
            category.method_path or "",
            category.source_file or "",
        )
        seen.setdefault(
            key,
            {
                "method_id": category.method_id or "",
                "method_name": category.method_name or "",
                "method_path": category.method_path or "",
                "source_file": category.source_file or "",
            },
        )
    return list(seen.values())


def _group_records(records: Sequence[CFAmbiguityRecord]) -> list[dict[str, str]]:
    grouped: dict[str, list[CFAmbiguityRecord]] = defaultdict(list)
    for record in records:
        key = record.group_key or record.ambiguity_key or f"{record.method_id}:{record.category_id}:{record.flow_id}:{record.process_id}:{record.exchange_id}:{record.exchange_index}"
        grouped[key].append(record)

    summaries: list[dict[str, str]] = []
    for group_key, group_records in sorted(grouped.items(), key=lambda item: (len(item[1]), item[0]), reverse=True):
        first = group_records[0]
        axes = sorted({axis.strip() for record in group_records for axis in record.differing_fields.split(",") if axis.strip()})
        issue_types = sorted({record.issue_type for record in group_records if record.issue_type})
        candidate_counts = [record.candidate_count for record in group_records]
        summaries.append(
            {
                "group_key": group_key,
                "issue_types": " | ".join(issue_types),
                "candidate_rows": str(len(group_records)),
                "candidate_count": str(max(candidate_counts) if candidate_counts else 0),
                "severity": max((record.severity for record in group_records), default=""),
                "ambiguity_key": first.ambiguity_key,
                "resolution_status": " | ".join(sorted({record.resolution_status for record in group_records if record.resolution_status})),
                "ambiguity_axes": " | ".join(axes) if axes else "unknown",
                "is_non_location_ambiguity": "true" if "location" not in axes else "false",
                "is_method_mixed": "true"
                if len({record.method_id for record in group_records if record.method_id}) > 1
                else "false",
                "method_id": first.method_id,
                "method_name": first.method_name,
                "category_id": first.category_id,
                "category_name": first.category_name,
                "flow_id": first.flow_id,
                "flow_name": first.flow_name,
                "process_id": first.process_id,
                "process_name": first.process_name,
                "exchange_id": first.exchange_id,
                "exchange_index": first.exchange_index,
                "candidate_method_ids": " | ".join(sorted({record.method_id for record in group_records if record.method_id})),
                "candidate_method_names": " | ".join(sorted({record.method_name for record in group_records if record.method_name})),
                "candidate_cf_values": " | ".join(sorted({record.cf_value for record in group_records if record.cf_value})),
                "candidate_source_files": " | ".join(sorted({record.source_file for record in group_records if record.source_file})),
                "candidate_flow_source_files": " | ".join(sorted({record.flow_source_file for record in group_records if record.flow_source_file})),
                "differing_fields": first.differing_fields,
                "message": first.message,
                "chosen_cf_value": " | ".join(sorted({record.chosen_cf_value for record in group_records if record.chosen_cf_value})),
                "rejected_cf_values": " | ".join(sorted({record.rejected_cf_values for record in group_records if record.rejected_cf_values})),
            }
        )
    return summaries


def _group_summaries_from_grouped_records(
    grouped: dict[str, list[CFAmbiguityRecord]],
) -> list[dict[str, str]]:
    summaries: list[dict[str, str]] = []
    for group_key, group_records in sorted(grouped.items(), key=lambda item: (len(item[1]), item[0]), reverse=True):
        first = group_records[0]
        axes = sorted({axis.strip() for record in group_records for axis in record.differing_fields.split(",") if axis.strip()})
        issue_types = sorted({record.issue_type for record in group_records if record.issue_type})
        candidate_counts = [record.candidate_count for record in group_records]
        summaries.append(
            {
                "group_key": group_key,
                "issue_types": " | ".join(issue_types),
                "candidate_rows": str(len(group_records)),
                "candidate_count": str(max(candidate_counts) if candidate_counts else 0),
                "severity": max((record.severity for record in group_records), default=""),
                "ambiguity_key": first.ambiguity_key,
                "resolution_status": " | ".join(sorted({record.resolution_status for record in group_records if record.resolution_status})),
                "ambiguity_axes": " | ".join(axes) if axes else "unknown",
                "is_non_location_ambiguity": "true" if "location" not in axes else "false",
                "is_method_mixed": "true"
                if len({record.method_id for record in group_records if record.method_id}) > 1
                else "false",
                "method_id": first.method_id,
                "method_name": first.method_name,
                "category_id": first.category_id,
                "category_name": first.category_name,
                "flow_id": first.flow_id,
                "flow_name": first.flow_name,
                "process_id": first.process_id,
                "process_name": first.process_name,
                "exchange_id": first.exchange_id,
                "exchange_index": first.exchange_index,
                "candidate_method_ids": " | ".join(sorted({record.method_id for record in group_records if record.method_id})),
                "candidate_method_names": " | ".join(sorted({record.method_name for record in group_records if record.method_name})),
                "candidate_cf_values": " | ".join(sorted({record.cf_value for record in group_records if record.cf_value})),
                "candidate_source_files": " | ".join(sorted({record.source_file for record in group_records if record.source_file})),
                "candidate_flow_source_files": " | ".join(sorted({record.flow_source_file for record in group_records if record.flow_source_file})),
                "differing_fields": first.differing_fields,
                "message": first.message,
                "chosen_cf_value": " | ".join(sorted({record.chosen_cf_value for record in group_records if record.chosen_cf_value})),
                "rejected_cf_values": " | ".join(sorted({record.rejected_cf_values for record in group_records if record.rejected_cf_values})),
            }
        )
    return summaries


def _summarise_records(records: Iterator[CFAmbiguityRecord]) -> tuple[Counter, Counter, list[dict[str, str]], int, int]:
    issue_type_counts = Counter()
    ambiguity_axis_counts = Counter()
    grouped: dict[str, list[CFAmbiguityRecord]] = defaultdict(list)
    non_location_record_count = 0
    method_mixed_record_count = 0

    for record in records:
        issue_type_counts[record.issue_type or "unknown"] += 1
        axes = [axis.strip() for axis in record.differing_fields.split(",") if axis.strip()]
        for axis in axes:
            ambiguity_axis_counts[axis] += 1
        if "location" not in set(axes):
            non_location_record_count += 1
        if len({str(item.get("method_id") or "") for item in _parse_candidate_metadata_json(record.all_candidate_metadata) if item.get("method_id")}) > 1:
            method_mixed_record_count += 1
        key = record.group_key or record.ambiguity_key or f"{record.method_id}:{record.category_id}:{record.flow_id}:{record.process_id}:{record.exchange_id}:{record.exchange_index}"
        grouped[key].append(record)

    return (
        issue_type_counts,
        ambiguity_axis_counts,
        _group_summaries_from_grouped_records(grouped),
        non_location_record_count,
        method_mixed_record_count,
    )


def _sorted_records(records: Sequence[CFAmbiguityRecord]) -> list[CFAmbiguityRecord]:
    return sorted(
        records,
        key=lambda record: (
            record.group_key or record.ambiguity_key,
            record.process_name,
            record.category_name,
            record.flow_name,
            record.candidate_index,
            record.severity,
        ),
    )


def explore_ambiguities(
    config: AmbiguityExploreConfig,
    *,
    progress_callback: Optional[ProgressCallback] = None,
) -> AmbiguityExploreResult:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "cf_ambiguities.csv"
    metadata_path = output_dir / "cf_ambiguities_metadata.json"

    warnings: List[WarningRecord] = []
    cf_ambiguities = _SpoolingAmbiguitySink()
    try:
        resolution_manager = CFResolutionManager(mode="cli")
        stats = CFAmbiguityStats()
        process_failures = 0
        run_started_at = time.monotonic()

        total_steps = 5
        current_step = 0
        _emit_progress(progress_callback, step="load_database", message="Step 1/5: Scanning database archive...", current=current_step, total=total_steps)
        database_archive = index_archive(config.database, require_processes=True, require_flows=True)
        current_step += 1
        _emit_progress(
            progress_callback,
            step="load_database",
            message=(
                f"Step 1/5: Indexed {len(database_archive.processes)} processes and {len(database_archive.flows)} flows."
            ),
            current=current_step,
            total=total_steps,
        )

        _emit_progress(progress_callback, step="load_methods", message="Step 2/5: Scanning optional methods input...", current=current_step, total=total_steps)
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
        current_step += 1
        _emit_progress(
            progress_callback,
            step="load_methods",
            message=(
                f"Step 2/5: Optional methods indexed."
                if methods_archive is not None
                else "Step 2/5: No external methods input provided."
            ),
            current=current_step,
            total=total_steps,
        )

        archives, lcia_method_source, internal_lcia_methods_ignored = resolve_lcia_archives(database_archive, methods_archive)
        active_unit_registry = merge_unit_registries(
            getattr(database_archive, "units", {}),
            getattr(methods_archive, "units", {}) if methods_archive is not None else {},
        )

        _emit_progress(
            progress_callback,
            step="collect_categories",
            message="Step 3/5: Collecting LCIA categories and CF candidates...",
            current=current_step,
            total=total_steps,
        )
        categories = collect_categories(
            archives,
            ambiguity_records=cf_ambiguities,
            diagnostic_file="cf_ambiguities_metadata.json",
        )
        selected_categories = list(select_lcia_categories(categories, config.method_selection))
        empty_selected_categories = list(ensure_category_factor_quality(selected_categories, warnings))
        if not selected_categories:
            raise LciReduceError("No LCIA categories selected for ambiguity exploration.")
        current_step += 1
        _emit_progress(
            progress_callback,
            step="collect_categories",
            message=f"Step 3/5: Selected {len(selected_categories)} LCIA categories for ambiguity exploration.",
            current=current_step,
            total=total_steps,
        )

        process_total = len(database_archive.processes)
        process_scan_limit = min(process_total, max(int(config.max_processes), 1))
        _emit_progress(
            progress_callback,
            step="scan_processes",
            message=f"Step 4/5: Scanning {process_scan_limit} of {process_total} processes for ambiguities...",
            current=current_step,
            total=total_steps,
            process_current=0,
            process_total=process_scan_limit,
        )
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

        process_index = 0
        last_progress_at = 0.0
        for locator, process_data in process_iterable:
            if process_index >= process_scan_limit:
                break
            process_index += 1
            process_name = str(process_data.get("name") or locator.name or locator.object_id)
            try:
                process_result = build_sparse_contribution_details(
                    exchanges=list(process_data.get("exchanges") or []),
                    flow_lookup=database_archive.flows,
                    categories=selected_categories,
                    unit_registry=active_unit_registry,
                    strict_units=config.strict_units,
                    tol=config.tolerance,
                    allow_water_mass_volume_override=config.allow_water_mass_volume_override,
                    process_data=process_data,
                    warning_records=warnings,
                    ambiguity_records=cf_ambiguities,
                    diagnostic_file="cf_ambiguities_metadata.json",
                    resolution_manager=resolution_manager,
                    explore_only=True,
                )
                stats.merge(process_result.cf_ambiguity_stats)
            except Exception as exc:
                process_failures += 1
                warnings.append(
                    _parse_warning_record(
                        source_name=database_archive.source_name,
                        source_file=locator.path,
                        message=str(exc),
                        input_role="ambiguity exploration",
                    )
                )

            now = time.monotonic()
            if process_index == 1 or process_index == process_scan_limit or now - last_progress_at >= 0.15:
                last_progress_at = now
                _emit_progress(
                    progress_callback,
                    step="scan_processes",
                    message=f"Step 4/5: Scanned {process_index}/{process_scan_limit} processes.",
                    current=current_step,
                    total=total_steps,
                    process_current=process_index,
                    process_total=process_scan_limit,
                    process_name=process_name,
                )

        current_step += 1
        _emit_progress(
            progress_callback,
            step="write_outputs",
            message="Step 5/5: Writing ambiguity outputs...",
            current=current_step,
            total=total_steps,
        )
        issue_type_counts, ambiguity_axis_counts, group_summaries, non_location_record_count, method_mixed_record_count = _summarise_records(
            cf_ambiguities.iter_records()
        )
        csv_export = write_cf_ambiguity_group_csv(
            csv_path,
            group_summaries,
            preferred_row_limit=_DEFAULT_AMBIGUITY_GROUP_CSV_ROW_LIMIT,
        )
        non_location_group_count = sum(1 for row in group_summaries if row["is_non_location_ambiguity"] == "true")
        method_mixed_group_count = sum(1 for row in group_summaries if row["is_method_mixed"] == "true")
        location_group_count = len(group_summaries) - non_location_group_count

        metadata = {
            "file_type": "cf_ambiguity_explorer",
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "scan_mode": "record_only_first_n_processes",
            "process_scan_limit": process_scan_limit,
            "source_format": database_archive.source_format,
            "database_name": database_archive.source_name,
            "database_hash": _hash_path(config.database),
            "lcia_source_hash": _hash_path(config.methods) if config.methods else "",
            "lcia_method_source": lcia_method_source,
            "internal_lcia_methods_ignored": internal_lcia_methods_ignored,
            "selected_impact_categories": [impact_category_report(category).__dict__ for category in selected_categories],
            "selected_methods": _selected_methods(selected_categories),
            "empty_selected_impact_categories": [
                impact_category_report(category).__dict__ for category in empty_selected_categories
            ],
            "unit_policy": "strict" if config.strict_units else "non_strict",
            "allow_water_mass_volume_override": config.allow_water_mass_volume_override,
            "csv_filename": csv_path.name,
            "csv_columns": list(_CF_AMBIGUITY_GROUP_CSV_FIELDS),
            "csv_row_mode": "group_summary",
            "csv_export_strategy": "largest-groups-first",
            "csv_rows_available": int(csv_export["rows_available"]),
            "csv_rows_written": int(csv_export["rows_written"]),
            "csv_truncated": bool(csv_export["truncated"]),
            "csv_row_limit_used": int(csv_export["row_limit_used"]),
            "n_processes_total": process_total,
            "n_processes_scanned": process_index,
            "n_processes_failed": process_failures,
            "n_cf_ambiguity_records": cf_ambiguities.count,
            "n_cf_ambiguity_groups": len(group_summaries),
            "n_non_location_ambiguity_records": non_location_record_count,
            "n_non_location_ambiguity_groups": non_location_group_count,
            "n_location_ambiguity_groups": location_group_count,
            "n_method_mixed_records": method_mixed_record_count,
            "n_method_mixed_groups": method_mixed_group_count,
            "issue_type_counts": dict(issue_type_counts),
            "ambiguity_axis_counts": dict(ambiguity_axis_counts),
            "cf_resolution_summary": resolution_manager.summary.__dict__,
            "cf_ambiguity_stats": stats.__dict__,
            "warnings": [warning.__dict__ for warning in warnings],
            "group_summaries": group_summaries,
            "runtime_seconds_total": time.monotonic() - run_started_at,
            "cf_ambiguities_csv": str(csv_path),
            "cf_ambiguity_metadata_json": str(metadata_path),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=True), encoding="utf-8")
        current_step += 1
        _emit_progress(
            progress_callback,
            step="write_outputs",
            message="Step 5/5: Ambiguity scan complete.",
            current=current_step,
            total=total_steps,
            process_current=process_index,
            process_total=process_scan_limit,
        )
        return AmbiguityExploreResult(
            cf_ambiguities_csv=str(csv_path),
            cf_ambiguity_metadata_json=str(metadata_path),
            metadata=metadata,
        )
    finally:
        cf_ambiguities.close()
