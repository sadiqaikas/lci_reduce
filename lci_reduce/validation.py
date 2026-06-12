"""Validation aggregation."""

from __future__ import annotations

import json
import errno
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from .cf_resolution import CFResolutionSummary
from .lcia import impact_category_report
from .manifest import write_manifest_csv
from .models import CFAmbiguityRecord, CFAmbiguityStats, ImpactCategory, ProcessReductionResult, WarningRecord


_LOCATION_DIFF_FIELDS = {"cf_location_id", "cf_location_name", "cf_region"}
_UNIT_DIFF_FIELDS = {"cf_unit", "cf_unit_id"}
_FLOW_PROPERTY_DIFF_FIELDS = {"cf_flow_property_id", "cf_flow_property_name"}
_COMPARTMENT_DIFF_FIELDS = {"cf_compartment", "cf_subcompartment"}

_CF_AMBIGUITY_CSV_FIELDS = [
    "severity",
    "issue_type",
    "ambiguity_axes",
    "is_non_location_ambiguity",
    "is_method_mixed",
    "group_key",
    "ambiguity_key",
    "resolution_status",
    "occurrence_timestamp",
    "candidate_selected",
    "candidate_index",
    "candidate_count",
    "method_id",
    "method_name",
    "candidate_method_ids",
    "candidate_method_names",
    "category_id",
    "category_name",
    "flow_id",
    "flow_name",
    "process_id",
    "process_name",
    "exchange_id",
    "exchange_index",
    "cf_value",
    "cf_unit",
    "cf_unit_id",
    "cf_flow_property_id",
    "cf_flow_property_name",
    "cf_compartment",
    "cf_subcompartment",
    "cf_location_id",
    "cf_location_name",
    "cf_region",
    "exchange_unit",
    "exchange_unit_id",
    "exchange_flow_property_id",
    "exchange_flow_property_name",
    "flow_reference_flow_property_id",
    "flow_reference_flow_property_name",
    "flow_source_file",
    "cf_source_file",
    "candidate_source_files",
    "candidate_cf_values",
    "candidate_cf_units",
    "candidate_cf_flow_property_ids",
    "candidate_cf_locations",
    "candidate_cf_regions",
    "candidate_cf_compartments",
    "candidate_cf_subcompartments",
    "candidate_flow_source_files",
    "differing_fields",
    "message",
    "all_candidate_cf_values",
    "all_candidate_metadata",
    "chosen_cf_value",
    "rejected_cf_values",
]


def _ordered_unique(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        token = value.strip()
        if not token or token in seen:
            continue
        seen.add(token)
        ordered.append(token)
    return ordered


def _join_unique(values: Sequence[str]) -> str:
    return " | ".join(_ordered_unique(values))


def _parse_candidate_metadata_json(text: str) -> list[dict]:
    if not text.strip():
        return []
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, dict)]


def _ambiguity_axes(differing_fields: str) -> list[str]:
    fields = {field.strip() for field in differing_fields.split(",") if field.strip()}
    axes: list[str] = []
    if fields & _UNIT_DIFF_FIELDS:
        axes.append("unit")
    if fields & _FLOW_PROPERTY_DIFF_FIELDS:
        axes.append("flow_property")
    if fields & _LOCATION_DIFF_FIELDS:
        axes.append("location")
    if fields & _COMPARTMENT_DIFF_FIELDS:
        axes.append("compartment")
    if not axes:
        axes.append("unknown")
    return axes


def _cf_ambiguity_csv_row(record: CFAmbiguityRecord) -> dict[str, str]:
    candidate_metadata = _parse_candidate_metadata_json(record.all_candidate_metadata)
    candidate_method_ids = _join_unique(str(item.get("method_id") or "") for item in candidate_metadata)
    candidate_method_names = _join_unique(str(item.get("method_name") or "") for item in candidate_metadata)
    candidate_source_files = _join_unique(str(item.get("source_file") or "") for item in candidate_metadata)
    candidate_cf_values = _join_unique(str(item.get("cf_value") or "") for item in candidate_metadata)
    candidate_cf_units = _join_unique(str(item.get("cf_unit") or "") for item in candidate_metadata)
    candidate_cf_flow_property_ids = _join_unique(str(item.get("cf_flow_property_id") or "") for item in candidate_metadata)
    candidate_cf_locations = _join_unique(
        str(item.get("cf_location_id") or item.get("cf_location_name") or "") for item in candidate_metadata
    )
    candidate_cf_regions = _join_unique(str(item.get("cf_region") or "") for item in candidate_metadata)
    candidate_cf_compartments = _join_unique(str(item.get("cf_compartment") or "") for item in candidate_metadata)
    candidate_cf_subcompartments = _join_unique(str(item.get("cf_subcompartment") or "") for item in candidate_metadata)
    candidate_flow_source_files = _join_unique(str(item.get("flow_source_file") or "") for item in candidate_metadata)
    axes = _ambiguity_axes(record.differing_fields)
    return {
        "severity": record.severity,
        "issue_type": record.issue_type,
        "ambiguity_axes": " | ".join(axes),
        "is_non_location_ambiguity": "true" if "location" not in axes else "false",
        "is_method_mixed": "true" if len(_ordered_unique(candidate_method_ids.split(" | "))) > 1 else "false",
        "group_key": record.group_key,
        "ambiguity_key": record.ambiguity_key,
        "resolution_status": record.resolution_status,
        "occurrence_timestamp": record.occurrence_timestamp,
        "candidate_selected": record.candidate_selected,
        "candidate_index": str(record.candidate_index),
        "candidate_count": str(record.candidate_count),
        "method_id": record.method_id,
        "method_name": record.method_name,
        "candidate_method_ids": candidate_method_ids,
        "candidate_method_names": candidate_method_names,
        "category_id": record.category_id,
        "category_name": record.category_name,
        "flow_id": record.flow_id,
        "flow_name": record.flow_name,
        "process_id": record.process_id,
        "process_name": record.process_name,
        "exchange_id": record.exchange_id,
        "exchange_index": record.exchange_index,
        "cf_value": record.cf_value,
        "cf_unit": record.cf_unit,
        "cf_unit_id": record.cf_unit_id,
        "cf_flow_property_id": record.cf_flow_property_id,
        "cf_flow_property_name": record.cf_flow_property_name,
        "cf_compartment": record.cf_compartment,
        "cf_subcompartment": record.cf_subcompartment,
        "cf_location_id": record.cf_location_id,
        "cf_location_name": record.cf_location_name,
        "cf_region": record.cf_region,
        "exchange_unit": record.exchange_unit,
        "exchange_unit_id": record.exchange_unit_id,
        "exchange_flow_property_id": record.exchange_flow_property_id,
        "exchange_flow_property_name": record.exchange_flow_property_name,
        "flow_reference_flow_property_id": record.flow_reference_flow_property_id,
        "flow_reference_flow_property_name": record.flow_reference_flow_property_name,
        "flow_source_file": record.flow_source_file,
        "cf_source_file": record.source_file,
        "candidate_source_files": candidate_source_files,
        "candidate_cf_values": candidate_cf_values,
        "candidate_cf_units": candidate_cf_units,
        "candidate_cf_flow_property_ids": candidate_cf_flow_property_ids,
        "candidate_cf_locations": candidate_cf_locations,
        "candidate_cf_regions": candidate_cf_regions,
        "candidate_cf_compartments": candidate_cf_compartments,
        "candidate_cf_subcompartments": candidate_cf_subcompartments,
        "candidate_flow_source_files": candidate_flow_source_files,
        "differing_fields": record.differing_fields,
        "message": record.message,
        "all_candidate_cf_values": record.all_candidate_cf_values,
        "all_candidate_metadata": record.all_candidate_metadata,
        "chosen_cf_value": record.chosen_cf_value,
        "rejected_cf_values": record.rejected_cf_values,
    }


def write_cf_ambiguities_csv(path: Path, cf_ambiguities: Sequence[CFAmbiguityRecord]) -> None:
    write_manifest_csv(path, (_cf_ambiguity_csv_row(record) for record in cf_ambiguities), _CF_AMBIGUITY_CSV_FIELDS)


_CF_AMBIGUITY_GROUP_CSV_FIELDS = [
    "severity",
    "issue_types",
    "ambiguity_axes",
    "is_non_location_ambiguity",
    "is_method_mixed",
    "group_key",
    "ambiguity_key",
    "resolution_status",
    "candidate_rows",
    "candidate_count",
    "method_id",
    "method_name",
    "category_id",
    "category_name",
    "flow_id",
    "flow_name",
    "process_id",
    "process_name",
    "exchange_id",
    "exchange_index",
    "candidate_method_ids",
    "candidate_method_names",
    "candidate_cf_values",
    "candidate_source_files",
    "candidate_flow_source_files",
    "differing_fields",
    "message",
    "chosen_cf_value",
    "rejected_cf_values",
]


def write_cf_ambiguity_group_csv(
    path: Path,
    group_rows: Sequence[dict[str, str]],
    *,
    preferred_row_limit: int = 10000,
    fallback_row_limits: Sequence[int] = (5000, 1000, 250, 50),
) -> dict[str, int | bool]:
    limits: list[int] = []
    for limit in (preferred_row_limit, *fallback_row_limits):
        limit_int = max(int(limit), 1)
        if limit_int not in limits:
            limits.append(limit_int)

    total_rows_available = len(group_rows)
    last_exc: OSError | None = None
    for limit in limits:
        rows_to_write = group_rows[: min(total_rows_available, limit)]
        try:
            write_manifest_csv(path, rows_to_write, _CF_AMBIGUITY_GROUP_CSV_FIELDS)
            return {
                "rows_written": len(rows_to_write),
                "rows_available": total_rows_available,
                "truncated": len(rows_to_write) < total_rows_available,
                "row_limit_used": limit,
            }
        except OSError as exc:
            last_exc = exc
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
            if exc.errno != errno.ENOSPC:
                raise

    if last_exc is not None:
        raise last_exc
    raise OSError(errno.ENOSPC, f"Failed to write ambiguity group CSV to {path}")


@dataclass
class ReductionRunSummary:
    n_processes_total: int = 0
    n_processes_modified: int = 0
    n_processes_unchanged: int = 0
    n_elementary_before: int = 0
    n_elementary_after: int = 0
    n_uncharacterised_total: int = 0
    n_uncharacterised_kept: int = 0
    n_uncharacterised_removed: int = 0
    n_coverage_failures: int = 0
    n_processes_positive_cover_ok: int = 0
    n_processes_negative_cover_ok: int = 0
    cf_ambiguity_stats: CFAmbiguityStats = field(default_factory=CFAmbiguityStats)

    def add_process_result(self, result: ProcessReductionResult) -> None:
        row = result.process_row
        self.n_processes_total += 1
        if row["status"] == "modified":
            self.n_processes_modified += 1
        else:
            self.n_processes_unchanged += 1
        self.n_elementary_before += int(row["n_elementary_before"])
        self.n_elementary_after += int(row["n_elementary_after"])
        self.n_uncharacterised_total += int(row["n_elementary_uncharacterised"])
        self.n_uncharacterised_kept += int(row.get("n_uncharacterised_kept", result.n_uncharacterised_kept))
        self.n_uncharacterised_removed += int(row.get("n_uncharacterised_removed", result.n_uncharacterised_removed))
        self.n_coverage_failures += int(bool(row["coverage_failure"]))
        self.n_processes_positive_cover_ok += int(bool(row["positive_cover_ok"]))
        self.n_processes_negative_cover_ok += int(bool(row["negative_cover_ok"]))
        self.cf_ambiguity_stats.merge(result.cf_ambiguity_stats)


def _summarise_results(reduced_results: Sequence[ProcessReductionResult]) -> ReductionRunSummary:
    summary = ReductionRunSummary()
    for result in reduced_results:
        summary.add_process_result(result)
    return summary


def build_run_summary(
    reduced_results: Sequence[ProcessReductionResult] | ReductionRunSummary,
    selected_categories: Sequence[ImpactCategory],
    empty_selected_categories: Sequence[ImpactCategory] | None,
    warnings: Sequence[WarningRecord],
    cf_ambiguities: Sequence[CFAmbiguityRecord],
    output_zip: str,
    exchange_manifest_csv: str,
    process_manifest_csv: str,
    config: dict,
    cf_resolution_summary: CFResolutionSummary | None = None,
) -> dict:
    def count_groups(issue_type: str) -> int:
        return len({record.group_key for record in cf_ambiguities if record.issue_type == issue_type and record.group_key})

    summary = reduced_results if isinstance(reduced_results, ReductionRunSummary) else _summarise_results(reduced_results)
    cf_resolution_summary = cf_resolution_summary or CFResolutionSummary()
    n_cf_unit_conflicts = count_groups("unit_conflict")
    n_cf_ambiguity_failures = count_groups("ambiguity_failure") + count_groups("duplicate_method_conflict")
    selected_category_rows = [impact_category_report(category).__dict__ for category in selected_categories]
    empty_category_rows = [
        impact_category_report(category).__dict__
        for category in (empty_selected_categories or [])
    ]
    return {
        "artifact_schema_version": 1,
        **config,
        "n_processes_total": summary.n_processes_total,
        "n_processes_modified": summary.n_processes_modified,
        "n_processes_unchanged": summary.n_processes_unchanged,
        "n_lcia_categories_used": len(selected_categories),
        "selected_lcia_categories": selected_category_rows,
        "n_empty_lcia_categories": len(empty_category_rows),
        "empty_lcia_categories": empty_category_rows,
        "n_elementary_before": summary.n_elementary_before,
        "n_elementary_after": summary.n_elementary_after,
        "n_elementary_removed": summary.n_elementary_before - summary.n_elementary_after,
        "n_uncharacterised_total": summary.n_uncharacterised_total,
        "n_uncharacterised_kept": summary.n_uncharacterised_kept,
        "n_uncharacterised_removed": summary.n_uncharacterised_removed,
        "n_coverage_failures": summary.n_coverage_failures,
        "n_unit_failures": n_cf_unit_conflicts,
        "n_water_mass_volume_overrides": sum(1 for warning in warnings if warning.object_type == "unit_override"),
        "n_ambiguous_mapping_failures": n_cf_ambiguity_failures,
        "n_missing_flow_failures": sum(1 for warning in warnings if warning.object_type == "flow"),
        "n_cf_duplicate_groups": count_groups("duplicate_deduplicated") + count_groups("duplicate_method_deduplicated"),
        "n_cf_duplicate_groups_deduplicated": count_groups("duplicate_deduplicated")
        + count_groups("duplicate_method_deduplicated"),
        "n_cf_ambiguity_failures": n_cf_ambiguity_failures,
        "n_cf_unit_conflicts": n_cf_unit_conflicts,
        "n_cf_regional_conflicts": count_groups("regional_conflict"),
        "n_cf_duplicate_method_conflicts": count_groups("duplicate_method_conflict"),
        "n_cf_ambiguities_found": cf_resolution_summary.n_cf_ambiguities_found,
        "n_cf_ambiguity_keys_unique": cf_resolution_summary.n_cf_ambiguity_keys_unique,
        "n_cf_ambiguities_resolved_automatically": cf_resolution_summary.n_cf_ambiguities_resolved_automatically,
        "n_cf_ambiguities_unresolved": cf_resolution_summary.n_cf_ambiguities_unresolved,
        "n_exact_cf_resolutions": summary.cf_ambiguity_stats.n_exact_cf_resolutions,
        "n_finite_cf_candidate_sets": summary.cf_ambiguity_stats.n_finite_cf_candidate_sets,
        "n_scenario_rows_added": summary.cf_ambiguity_stats.n_scenario_rows_added,
        "n_regional_scenario_groups": summary.cf_ambiguity_stats.n_regional_scenario_groups,
        "n_independent_candidate_groups": summary.cf_ambiguity_stats.n_independent_candidate_groups,
        "n_unresolved_cf_ambiguities": summary.cf_ambiguity_stats.n_unresolved_cf_ambiguities,
        "max_candidate_set_size": summary.cf_ambiguity_stats.max_candidate_set_size,
        "n_processes_positive_cover_ok": summary.n_processes_positive_cover_ok,
        "n_processes_negative_cover_ok": summary.n_processes_negative_cover_ok,
        "output_zip": output_zip,
        "exchange_manifest_csv": exchange_manifest_csv,
        "process_manifest_csv": process_manifest_csv,
        "n_warning_records": len(warnings),
        "n_cf_ambiguity_records": len(cf_ambiguities),
        "warnings": [warning.__dict__ for warning in warnings],
    }


def build_validation_report(
    reduced_results: Sequence[ProcessReductionResult] | ReductionRunSummary,
    selected_categories: Sequence[ImpactCategory],
    empty_selected_categories: Sequence[ImpactCategory] | None,
    warnings: Sequence[WarningRecord],
    cf_ambiguities: Sequence[CFAmbiguityRecord],
    output_zip: str,
    pdf_report: str = "",
    cf_resolution_summary: CFResolutionSummary | None = None,
) -> dict:
    report = build_run_summary(
        reduced_results=reduced_results,
        selected_categories=selected_categories,
        empty_selected_categories=empty_selected_categories,
        warnings=warnings,
        cf_ambiguities=cf_ambiguities,
        output_zip=output_zip,
        exchange_manifest_csv="",
        process_manifest_csv="",
        config={},
        cf_resolution_summary=cf_resolution_summary,
    )
    return report
