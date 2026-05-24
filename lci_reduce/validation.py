"""Validation aggregation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .cf_resolution import CFResolutionChoiceRecord, CFResolutionSummary
from .lcia import impact_category_report
from .models import CFAmbiguityRecord, ImpactCategory, ProcessReductionResult, WarningRecord


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
    cf_resolution_choices: Sequence[CFResolutionChoiceRecord] | None = None,
    cf_resolution_summary: CFResolutionSummary | None = None,
) -> dict:
    def count_groups(issue_type: str) -> int:
        return len({record.group_key for record in cf_ambiguities if record.issue_type == issue_type and record.group_key})

    summary = reduced_results if isinstance(reduced_results, ReductionRunSummary) else _summarise_results(reduced_results)
    cf_resolution_summary = cf_resolution_summary or CFResolutionSummary()
    cf_resolution_choices = list(cf_resolution_choices or [])
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
        "n_cf_unique_user_decisions": cf_resolution_summary.n_cf_unique_user_decisions,
        "n_cf_ambiguities_resolved_by_user_choice": cf_resolution_summary.n_cf_ambiguities_resolved_by_user_choice,
        "n_cf_user_decision_applications": cf_resolution_summary.n_cf_ambiguities_resolved_by_user_choice,
        "n_cf_ambiguities_unresolved": cf_resolution_summary.n_cf_ambiguities_unresolved,
        "n_cf_resolution_choices_reused": cf_resolution_summary.n_cf_resolution_choices_reused,
        "n_cf_user_decision_reuses": cf_resolution_summary.n_cf_resolution_choices_reused,
        "n_processes_positive_cover_ok": summary.n_processes_positive_cover_ok,
        "n_processes_negative_cover_ok": summary.n_processes_negative_cover_ok,
        "output_zip": output_zip,
        "exchange_manifest_csv": exchange_manifest_csv,
        "process_manifest_csv": process_manifest_csv,
        "n_warning_records": len(warnings),
        "n_cf_ambiguity_records": len(cf_ambiguities),
        "n_cf_resolution_choice_records": len(cf_resolution_choices),
        "warnings": [warning.__dict__ for warning in warnings],
        "cf_ambiguities": [
            {
                "severity": record.severity,
                "method_id": record.method_id,
                "method_name": record.method_name,
                "category_id": record.category_id,
                "category_name": record.category_name,
                "flow_id": record.flow_id,
                "flow_name": record.flow_name,
                "candidate_count": record.candidate_count,
                "candidate_index": record.candidate_index,
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
                "source_file": record.source_file,
                "differing_fields": record.differing_fields,
                "message": record.message,
                "issue_type": record.issue_type,
                "group_key": record.group_key,
                "process_id": record.process_id,
                "process_name": record.process_name,
                "exchange_id": record.exchange_id,
                "exchange_index": record.exchange_index,
                "ambiguity_key": record.ambiguity_key,
                "resolution_status": record.resolution_status,
                "choice_origin": record.choice_origin,
                "occurrence_timestamp": record.occurrence_timestamp,
                "all_candidate_cf_values": record.all_candidate_cf_values,
                "all_candidate_metadata": record.all_candidate_metadata,
                "chosen_cf_value": record.chosen_cf_value,
                "rejected_cf_values": record.rejected_cf_values,
                "candidate_selected": record.candidate_selected,
            }
            for record in cf_ambiguities
        ],
        "cf_resolution_choices": [record.__dict__ for record in cf_resolution_choices],
    }


def build_validation_report(
    reduced_results: Sequence[ProcessReductionResult] | ReductionRunSummary,
    selected_categories: Sequence[ImpactCategory],
    empty_selected_categories: Sequence[ImpactCategory] | None,
    warnings: Sequence[WarningRecord],
    cf_ambiguities: Sequence[CFAmbiguityRecord],
    output_zip: str,
    pdf_report: str = "",
    cf_ambiguities_csv: str = "",
    cf_resolution_choices_csv: str = "",
    cf_resolution_summary: CFResolutionSummary | None = None,
) -> dict:
    del pdf_report, cf_ambiguities_csv
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
        cf_resolution_choices=[],
        cf_resolution_summary=cf_resolution_summary,
    )
    report["cf_resolution_choices_csv"] = cf_resolution_choices_csv
    return report
