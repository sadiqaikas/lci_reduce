from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from lci_reduce.priority_analyser import (
    PriorityAnalysisError,
    analyse_selected_group,
    build_priority_summary,
    filter_priority_rows,
    load_priority_dataset,
    match_mixed_flow_items,
    rank_rows_by_eta,
)


DEFAULT_FIELDNAMES = [
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
    "eta_0_95",
    "loss_max_0_95",
    "eta_0_99",
    "loss_max_0_99",
    "cf_status",
]


def _priority_row(**overrides: str) -> dict[str, str]:
    row = {
        "flow_id": "flow-default",
        "flow_name": "Default flow",
        "compartment": "air",
        "subcompartment": "urban air",
        "reference_unit": "kg",
        "occurrence_count": "1",
        "characterised_occurrence_count": "1",
        "tau_entry_min": "0.5",
        "tau_entry_median": "0.5",
        "tau_entry_max": "0.5",
        "eta_0_95": "0",
        "loss_max_0_95": "0",
        "eta_0_99": "0",
        "loss_max_0_99": "0",
        "cf_status": "characterised",
    }
    row.update(overrides)
    return row


def _write_priority_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str] | None = None) -> Path:
    actual_fieldnames = fieldnames or DEFAULT_FIELDNAMES
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=actual_fieldnames)
        writer.writeheader()
        writer.writerows(
            [{field: row.get(field, "") for field in actual_fieldnames} for row in rows]
        )
    return path


def test_single_flow_ranking_uses_eta_then_loss_max(tmp_path: Path) -> None:
    csv_path = _write_priority_csv(
        tmp_path / "priority.csv",
        [
            _priority_row(flow_id="A", flow_name="A", eta_0_95="0.5", loss_max_0_95="0.6"),
            _priority_row(flow_id="B", flow_name="B", eta_0_95="0.2", loss_max_0_95="0.9"),
            _priority_row(flow_id="C", flow_name="C", eta_0_95="0.0", loss_max_0_95="0.1"),
        ],
    )
    dataset = load_priority_dataset(csv_path)
    pair = dataset.get_tau_pair(0.95)
    ranked = rank_rows_by_eta(dataset.rows, pair)
    assert [row.flow_id for row in ranked] == ["A", "B", "C"]


def test_group_bound_does_not_sum_eta_values(tmp_path: Path) -> None:
    csv_path = _write_priority_csv(
        tmp_path / "priority.csv",
        [
            _priority_row(flow_id="A", flow_name="A", eta_0_95="0.0", loss_max_0_95="0.02"),
            _priority_row(flow_id="B", flow_name="B", eta_0_95="0.0", loss_max_0_95="0.02"),
        ],
    )
    dataset = load_priority_dataset(csv_path)
    pair = dataset.get_tau_pair(0.95)
    result = analyse_selected_group(dataset.rows, pair)
    assert result.lower_bound == pytest.approx(0.0)
    assert result.upper_bound == pytest.approx(0.04)
    assert result.sum_loss_max == pytest.approx(0.04)
    assert result.exact_eta is None


def test_group_bound_with_serious_lower_bound(tmp_path: Path) -> None:
    csv_path = _write_priority_csv(
        tmp_path / "priority.csv",
        [
            _priority_row(flow_id="A", flow_name="A", eta_0_95="0.42", loss_max_0_95="0.50"),
            _priority_row(flow_id="B", flow_name="B", eta_0_95="0.08", loss_max_0_95="0.13"),
        ],
    )
    dataset = load_priority_dataset(csv_path)
    pair = dataset.get_tau_pair(0.95)
    result = analyse_selected_group(dataset.rows, pair)
    assert result.lower_bound == pytest.approx(0.42)
    assert result.upper_bound == pytest.approx(0.63)
    assert result.exact_eta is None


def test_single_selected_flow_reports_exact_eta(tmp_path: Path) -> None:
    csv_path = _write_priority_csv(
        tmp_path / "priority.csv",
        [
            _priority_row(flow_id="A", flow_name="A", eta_0_95="0.42", loss_max_0_95="0.50"),
            _priority_row(flow_id="B", flow_name="B", eta_0_95="0.08", loss_max_0_95="0.13"),
        ],
    )
    dataset = load_priority_dataset(csv_path)
    pair = dataset.get_tau_pair(0.95)
    result = analyse_selected_group([dataset.rows_by_flow_id["B"]], pair)
    assert result.lower_bound == pytest.approx(0.08)
    assert result.upper_bound == pytest.approx(0.13)
    assert result.exact_eta == pytest.approx(0.08)
    assert result.exact_reason is not None


def test_group_bound_collapsing_to_zero_reports_exact_zero(tmp_path: Path) -> None:
    csv_path = _write_priority_csv(
        tmp_path / "priority.csv",
        [
            _priority_row(flow_id="A", flow_name="A", eta_0_95="0", loss_max_0_95="0"),
            _priority_row(flow_id="B", flow_name="B", eta_0_95="0", loss_max_0_95="0"),
        ],
    )
    dataset = load_priority_dataset(csv_path)
    pair = dataset.get_tau_pair(0.95)
    result = analyse_selected_group(dataset.rows, pair)
    assert result.lower_bound == pytest.approx(0.0)
    assert result.upper_bound == pytest.approx(0.0)
    assert result.exact_eta == pytest.approx(0.0)
    assert result.exact_reason is not None


def test_upper_bound_is_capped_at_tau(tmp_path: Path) -> None:
    csv_path = _write_priority_csv(
        tmp_path / "priority.csv",
        [
            _priority_row(flow_id="A", flow_name="A", eta_0_95="0.95", loss_max_0_95="1.0"),
            _priority_row(flow_id="B", flow_name="B", eta_0_95="0.20", loss_max_0_95="0.50"),
        ],
    )
    dataset = load_priority_dataset(csv_path)
    pair = dataset.get_tau_pair(0.95)
    result = analyse_selected_group(dataset.rows, pair)
    assert result.lower_bound == pytest.approx(0.95)
    assert result.upper_bound == pytest.approx(0.95)
    assert result.capped is True


def test_group_risk_only_filter(tmp_path: Path) -> None:
    csv_path = _write_priority_csv(
        tmp_path / "priority.csv",
        [
            _priority_row(flow_id="A", flow_name="A", eta_0_95="0", loss_max_0_95="0.05"),
            _priority_row(flow_id="B", flow_name="B", eta_0_95="0", loss_max_0_95="0"),
            _priority_row(flow_id="C", flow_name="C", eta_0_95="0.1", loss_max_0_95="0.2"),
        ],
    )
    dataset = load_priority_dataset(csv_path)
    pair = dataset.get_tau_pair(0.95)
    filtered = filter_priority_rows(dataset, pair, group_risk_only=True)
    assert [row.flow_id for row in filtered] == ["A"]


def test_critical_only_at_099_filter(tmp_path: Path) -> None:
    csv_path = _write_priority_csv(
        tmp_path / "priority.csv",
        [
            _priority_row(flow_id="A", flow_name="A", eta_0_95="0", eta_0_99="0.04"),
            _priority_row(flow_id="B", flow_name="B", eta_0_95="0.01", eta_0_99="0.04"),
            _priority_row(flow_id="C", flow_name="C", eta_0_95="0", eta_0_99="0"),
        ],
    )
    dataset = load_priority_dataset(csv_path)
    pair = dataset.get_tau_pair(0.99)
    filtered = filter_priority_rows(dataset, pair, critical_only_at_099=True)
    assert [row.flow_id for row in filtered] == ["A"]


def test_schema_validation_and_custom_tau_detection(tmp_path: Path) -> None:
    broken_path = _write_priority_csv(
        tmp_path / "broken.csv",
        [
            _priority_row(flow_id="A", flow_name="A"),
        ],
        fieldnames=[
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
            "eta_0_95",
            "cf_status",
        ],
    )
    with pytest.raises(PriorityAnalysisError, match="missing one side of an eta/loss_max pair"):
        load_priority_dataset(broken_path)

    custom_path = _write_priority_csv(
        tmp_path / "custom.csv",
        [
            {
                "flow_id": "A",
                "flow_name": "A",
                "compartment": "air",
                "subcompartment": "urban air",
                "reference_unit": "kg",
                "occurrence_count": "1",
                "characterised_occurrence_count": "1",
                "tau_entry_min": "0.1",
                "tau_entry_median": "0.1",
                "tau_entry_max": "0.1",
                "eta_0_975": "0.2",
                "loss_max_0_975": "0.3",
                "cf_status": "characterised",
            }
        ],
        fieldnames=[
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
            "eta_0_975",
            "loss_max_0_975",
            "cf_status",
        ],
    )
    dataset = load_priority_dataset(custom_path)
    assert dataset.get_tau_pair(None).tau == pytest.approx(0.975)


def test_name_matching_and_flow_id_priority(tmp_path: Path) -> None:
    csv_path = _write_priority_csv(
        tmp_path / "priority.csv",
        [
            _priority_row(flow_id="id-takes-priority", flow_name="Different name"),
            _priority_row(flow_id="flow-2", flow_name="Exact Name"),
            _priority_row(flow_id="flow-3", flow_name="Case Unique"),
            _priority_row(flow_id="flow-4", flow_name="AMBIG"),
            _priority_row(flow_id="flow-5", flow_name="ambig"),
        ],
    )
    dataset = load_priority_dataset(csv_path)
    result = match_mixed_flow_items(
        dataset,
        ["id-takes-priority", "Exact Name", "case unique", "AmBiG", "missing"],
    )
    assert result.matched_flow_ids == ["id-takes-priority", "flow-2", "flow-3"]
    assert result.unmatched_items == ["missing"]
    assert "AmBiG" in result.ambiguous_items
    assert len(result.ambiguous_items["AmBiG"]) == 2


def test_null_tau_entry_values_sort_last(tmp_path: Path) -> None:
    csv_path = _write_priority_csv(
        tmp_path / "priority.csv",
        [
            _priority_row(flow_id="A", flow_name="A", eta_0_95="0.1", loss_max_0_95="0.1", tau_entry_min="0.1"),
            _priority_row(flow_id="B", flow_name="B", eta_0_95="0.1", loss_max_0_95="0.1", tau_entry_min=""),
            _priority_row(flow_id="C", flow_name="C", eta_0_95="0.1", loss_max_0_95="0.1", tau_entry_min="0.05"),
        ],
    )
    dataset = load_priority_dataset(csv_path)
    pair = dataset.get_tau_pair(0.95)
    ranked = rank_rows_by_eta(dataset.rows, pair)
    assert [row.flow_id for row in ranked] == ["C", "A", "B"]


def test_loading_and_analysing_priority_file_does_not_modify_it(tmp_path: Path) -> None:
    csv_path = _write_priority_csv(
        tmp_path / "priority.csv",
        [
            _priority_row(flow_id="A", flow_name="A", eta_0_95="0.2", loss_max_0_95="0.3"),
        ],
    )
    original_bytes = csv_path.read_bytes()
    dataset = load_priority_dataset(csv_path)
    summary = build_priority_summary(dataset, dataset.get_tau_pair(0.95), top_n=10, selected_rows=dataset.rows)
    assert summary["selected_flow_analysis"]["lower_bound_eta"] == pytest.approx(0.2)
    assert csv_path.read_bytes() == original_bytes


def test_analyse_priority_cli_outputs_summary_and_ranked_csv(tmp_path: Path) -> None:
    csv_path = _write_priority_csv(
        tmp_path / "priority.csv",
        [
            _priority_row(flow_id="A", flow_name="A", eta_0_95="0.5", loss_max_0_95="0.6"),
            _priority_row(flow_id="B", flow_name="B", eta_0_95="0.0", loss_max_0_95="0.02"),
        ],
    )
    summary_json = tmp_path / "summary.json"
    ranked_csv = tmp_path / "ranked.csv"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd())

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "lci_reduce.cli",
            "analyse-priority",
            "--priority-csv",
            str(csv_path),
            "--top-n",
            "5",
            "--select-flow-id",
            "A",
            "--output-summary-json",
            str(summary_json),
            "--output-ranked-csv",
            str(ranked_csv),
        ],
        cwd=Path.cwd(),
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "Priority File Analyser" in result.stdout
    assert "Top 5 By Eta" in result.stdout
    assert "Compact-Screen Group Bound" in result.stdout
    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert summary["selected_flow_analysis"]["lower_bound_eta"] == pytest.approx(0.5)
    assert ranked_csv.exists()
