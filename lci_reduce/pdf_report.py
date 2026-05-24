"""PDF report generation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Sequence

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

from .models import CreateConfig, ImpactCategory, WarningRecord


def _paragraphs(text_lines: Iterable[str], style):
    for line in text_lines:
        yield Paragraph(line, style)
        yield Spacer(1, 6)


def write_pdf_report(
    pdf_path: Path,
    input_zip: str,
    methods_input: str,
    output_zip: str,
    config: CreateConfig,
    selected_categories: Sequence[ImpactCategory],
    validation: dict,
    warning_records: Sequence[WarningRecord],
    output_files: Dict[str, str],
) -> None:
    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4)
    styles = getSampleStyleSheet()
    body = styles["BodyText"]
    heading = styles["Heading1"]
    story = [Paragraph("lci_reduce Database Reduction Report", heading), Spacer(1, 12)]

    input_lines = [
        f"Input ZIP: {input_zip}",
        f"Optional methods input: {methods_input or 'None'}",
        f"Output ZIP: {output_zip}",
        f"Tau: {config.tau}",
        f"LCIA selection mode: {config.method_selection}",
        f"Selected LCIA categories: {len(selected_categories)}",
        f"Empty selected LCIA categories: {validation.get('n_empty_lcia_categories', 0)}",
        f"Uncharacterised policy: {config.uncharacterised_policy}",
        f"Strict units: {config.strict_units}",
    ]
    story.extend(_paragraphs(input_lines, body))

    before = validation["n_elementary_before"]
    after = validation["n_elementary_after"]
    removed = validation["n_elementary_removed"]
    percent_removed = (removed / before * 100.0) if before else 0.0
    database_lines = [
        f"Processes: {validation['n_processes_total']}",
        f"Elementary exchanges before: {before}",
        f"Elementary exchanges after: {after}",
        f"Elementary exchanges removed: {removed}",
        f"Percentage removed: {percent_removed:.2f}%",
        (
            "Uncharacterised elementary exchanges: "
            f"total={validation['n_uncharacterised_total']}, "
            f"kept={validation['n_uncharacterised_kept']}, "
            f"removed={validation['n_uncharacterised_removed']}"
        ),
    ]
    story.extend(_paragraphs(database_lines, body))

    validation_lines = [
        f"Processes with successful positive coverage: {validation['n_processes_positive_cover_ok']}",
        f"Processes with successful negative coverage: {validation['n_processes_negative_cover_ok']}",
        f"Coverage failures: {validation['n_coverage_failures']}",
        "Positive and negative characterised contributions were covered separately.",
    ]
    story.extend(_paragraphs(validation_lines, body))

    warning_lines = [
        f"CF ambiguities found: {validation.get('n_cf_ambiguities_found', 0)}",
        f"Unique CF ambiguity keys: {validation.get('n_cf_ambiguity_keys_unique', 0)}",
        f"CF ambiguities resolved automatically: {validation.get('n_cf_ambiguities_resolved_automatically', 0)}",
        f"Unique user decisions: {validation.get('n_cf_unique_user_decisions', 0)}",
        f"User-decision applications: {validation.get('n_cf_user_decision_applications', 0)}",
        f"User-decision reuses: {validation.get('n_cf_resolution_choices_reused', 0)}",
        f"CF ambiguities unresolved: {validation.get('n_cf_ambiguities_unresolved', 0)}",
        f"CF ambiguity failures: {validation.get('n_cf_ambiguity_failures', 0)}",
        f"CF unit conflicts: {validation.get('n_cf_unit_conflicts', 0)}",
        f"CF regional conflicts: {validation.get('n_cf_regional_conflicts', 0)}",
        f"Duplicate LCIA method conflicts: {validation.get('n_cf_duplicate_method_conflicts', 0)}",
    ]
    serious_warnings = [
        warning.message for warning in warning_records if warning.severity.lower() in {"error", "warning"}
    ][:6]
    warning_lines.extend(serious_warnings or ["No serious warnings recorded."])
    story.extend(_paragraphs(warning_lines, body))

    selected_category_lines = ["Selected LCIA category details:"]
    for row in validation.get("selected_lcia_categories", []):
        selected_category_lines.append(
            " | ".join(
                [
                    row.get("method_name") or "-",
                    row.get("category_name") or "-",
                    row.get("method_id") or "-",
                    row.get("method_path") or "-",
                    row.get("source_file") or "-",
                ]
            )
        )
    story.extend(_paragraphs(selected_category_lines or ["Selected LCIA category details: none"], body))

    empty_category_lines = [
        f"Empty selected LCIA categories: {validation.get('n_empty_lcia_categories', 0)}"
    ]
    for row in validation.get("empty_lcia_categories", []):
        empty_category_lines.append(
            " | ".join(
                [
                    row.get("method_name") or "-",
                    row.get("category_name") or "-",
                    row.get("method_id") or "-",
                    row.get("source_file") or "-",
                ]
            )
        )
    story.extend(_paragraphs(empty_category_lines, body))

    output_lines = [
        f"Lite ZIP path: {output_files['lite_zip']}",
        f"Exchange manifest: {output_files['manifest_exchanges']}",
        f"Process manifest: {output_files['manifest_processes']}",
        f"Validation JSON: {output_files['validation_report']}",
        f"Warnings CSV: {output_files['warnings_csv']}",
        f"CF ambiguities CSV: {output_files.get('cf_ambiguities_csv') or 'Not created'}",
        f"CF resolution choices CSV: {output_files.get('cf_resolution_choices_csv') or 'Not created'}",
    ]
    story.extend(_paragraphs(output_lines, body))
    doc.build(story)
