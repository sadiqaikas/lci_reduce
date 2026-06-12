from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from lca_flowkit.analysis import analyse_flows
from lca_flowkit.errors import AnalysisError


def _write_priority_csv(path: Path) -> Path:
    rows = [
        {
            "flow_id": "A",
            "flow_name": "Alpha",
            "compartment": "air",
            "subcompartment": "urban air",
            "reference_unit": "kg",
            "occurrence_count": "1",
            "characterised_occurrence_count": "1",
            "tau_entry_min": "0.4",
            "tau_entry_median": "0.4",
            "tau_entry_max": "0.4",
            "eta_0_95": "0.11",
            "eta_0_95_witness": "P1 | Climate change | pos",
            "loss_max_0_95": "0.15",
        },
        {
            "flow_id": "B",
            "flow_name": "Beta",
            "compartment": "air",
            "subcompartment": "urban air",
            "reference_unit": "kg",
            "occurrence_count": "1",
            "characterised_occurrence_count": "1",
            "tau_entry_min": "0.5",
            "tau_entry_median": "0.5",
            "tau_entry_max": "0.5",
            "eta_0_95": "0.05",
            "eta_0_95_witness": "P2 | Climate change | pos",
            "loss_max_0_95": "0.06",
        },
        {
            "flow_id": "C",
            "flow_name": "Shared",
            "compartment": "water",
            "subcompartment": "fresh water",
            "reference_unit": "kg",
            "occurrence_count": "1",
            "characterised_occurrence_count": "1",
            "tau_entry_min": "0.6",
            "tau_entry_median": "0.6",
            "tau_entry_max": "0.6",
            "eta_0_95": "0.02",
            "eta_0_95_witness": "P3 | Eutrophication | neg",
            "loss_max_0_95": "0.08",
        },
        {
            "flow_id": "D",
            "flow_name": "Shared",
            "compartment": "water",
            "subcompartment": "fresh water",
            "reference_unit": "kg",
            "occurrence_count": "1",
            "characterised_occurrence_count": "1",
            "tau_entry_min": "0.7",
            "tau_entry_median": "0.7",
            "tau_entry_max": "0.7",
            "eta_0_95": "0.01",
            "eta_0_95_witness": "P4 | Eutrophication | neg",
            "loss_max_0_95": "0.02",
        },
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_grouped_flow_analyser_bounds_and_warning(tmp_path: Path) -> None:
    csv_path = _write_priority_csv(tmp_path / "priority.csv")
    metadata_path = tmp_path / "priority.json"
    metadata_path.write_text(json.dumps({"audit_tau": [0.95]}), encoding="utf-8")
    report = analyse_flows(
        str(csv_path),
        str(metadata_path),
        flow_ids=["A", "B"],
        tau=0.95,
        top_n=5,
    )
    assert report.eta_bounds["lower_bound_eta"] == pytest.approx(0.11)
    assert report.eta_bounds["upper_bound_eta"] == pytest.approx(0.17)
    assert report.eta_bounds["sum_loss_max"] == pytest.approx(0.21)
    assert report.eta_bounds["exact_eta"] is None
    assert report.warnings


def test_analyser_reports_ambiguous_name_matches(tmp_path: Path) -> None:
    csv_path = _write_priority_csv(tmp_path / "priority.csv")
    report = analyse_flows(str(csv_path), flow_names=["Shared"], tau=0.95)
    assert report.matched_flow_ids == []
    assert report.ambiguous_names == {"Shared": ["C", "D"]}


def test_all_flows_selects_entire_priority_file(tmp_path: Path) -> None:
    csv_path = _write_priority_csv(tmp_path / "priority.csv")
    report = analyse_flows(str(csv_path), all_flows=True, tau=0.95)
    assert report.matched_flow_ids == ["A", "B", "C", "D"]
    assert report.unmatched_flow_ids == []
    assert report.unmatched_flow_names == []
    assert report.ambiguous_names == {}


def test_all_flows_rejects_explicit_selectors(tmp_path: Path) -> None:
    csv_path = _write_priority_csv(tmp_path / "priority.csv")
    with pytest.raises(AnalysisError, match="all_flows=True cannot be combined"):
        analyse_flows(str(csv_path), flow_ids=["A"], all_flows=True, tau=0.95)
    with pytest.raises(AnalysisError, match="all_flows=True cannot be combined"):
        analyse_flows(str(csv_path), flow_names=["Alpha"], all_flows=True, tau=0.95)


def test_analyser_requires_any_selection_mode(tmp_path: Path) -> None:
    csv_path = _write_priority_csv(tmp_path / "priority.csv")
    with pytest.raises(AnalysisError, match="Provide flow_ids or flow_names, or set all_flows=True"):
        analyse_flows(str(csv_path), tau=0.95)


def test_result_types_filters_analysis_sections(tmp_path: Path) -> None:
    csv_path = _write_priority_csv(tmp_path / "priority.csv")
    report = analyse_flows(
        str(csv_path),
        flow_ids=["A", "B"],
        tau=0.95,
        result_types=["loss_max_flows", "coverage_summary"],
    )
    assert report.requested_result_types == ["loss_max_flows", "coverage_summary"]
    assert report.loss_max_flows is not None
    assert report.selected_flows == report.loss_max_flows
    assert report.coverage_summary is not None
    assert report.eta_bounds is None
    assert report.witness_processes is None
    assert report.top_n_repair is None
    assert report.least_n_repair is None


def test_result_types_rejects_unknown_section(tmp_path: Path) -> None:
    csv_path = _write_priority_csv(tmp_path / "priority.csv")
    with pytest.raises(AnalysisError, match="Unsupported result type"):
        analyse_flows(str(csv_path), flow_ids=["A"], tau=0.95, result_types=["not_a_section"])
