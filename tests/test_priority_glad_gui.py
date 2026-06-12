import csv
import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

from lci_reduce import priority_glad as priority_glad_module
from lci_reduce.priority_glad_gui import PriorityGladPanel


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _write_priority_csv(path: Path) -> Path:
    fieldnames = [
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
    ]
    rows = [
        {
            "flow_id": "flow-a",
            "flow_name": "Alpha",
            "compartment": "air",
            "subcompartment": "urban air",
            "reference_unit": "kg",
            "occurrence_count": "2",
            "characterised_occurrence_count": "2",
            "tau_entry_min": "0.1",
            "tau_entry_median": "0.2",
            "tau_entry_max": "0.3",
            "eta_0_95": "0",
            "loss_max_0_95": "0",
            "eta_0_99": "0.1",
            "loss_max_0_99": "0.1",
        },
        {
            "flow_id": "flow-b",
            "flow_name": "Beta",
            "compartment": "air",
            "subcompartment": "urban air",
            "reference_unit": "kg",
            "occurrence_count": "2",
            "characterised_occurrence_count": "2",
            "tau_entry_min": "0.4",
            "tau_entry_median": "0.5",
            "tau_entry_max": "0.6",
            "eta_0_95": "0.2",
            "loss_max_0_95": "0.5",
            "eta_0_99": "0.2",
            "loss_max_0_99": "0.5",
        },
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_snapshot(path: Path, list_name: str) -> None:
    fieldnames = [
        "source_file",
        "list_name",
        "flow_uuid",
        "flow_name",
        "context",
        "unit",
        "cas_number",
        "synonyms",
    ]
    rows = [
        {
            "source_file": f"{list_name}.csv",
            "list_name": list_name,
            "flow_uuid": "flow-a",
            "flow_name": "Alpha",
            "context": "air/urban air",
            "unit": "kg",
            "cas_number": "",
            "synonyms": "",
        },
        {
            "source_file": f"{list_name}.csv",
            "list_name": list_name,
            "flow_uuid": "uuid-beta",
            "flow_name": "Beta",
            "context": "air/urban air",
            "unit": "kg",
            "cas_number": "",
            "synonyms": "",
        },
        {
            "source_file": f"{list_name}.csv",
            "list_name": list_name,
            "flow_uuid": "uuid-beta-2",
            "flow_name": "Beta",
            "context": "air/urban air",
            "unit": "kg",
            "cas_number": "",
            "synonyms": "",
        },
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_asset_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    manifest = {"upstream_repo_url": "https://example.invalid/glad", "targets": {}}
    for name in ("ecoinventEFv3.7", "ILCD_EFv3.0", "FEDEFLv1.0.3", "IDEA_EFv2.3"):
        _write_snapshot(path / f"{name}.csv", name)
        manifest["targets"][name] = {"snapshot": f"{name}.csv"}
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _patch_assets(monkeypatch: pytest.MonkeyPatch, asset_dir: Path) -> None:
    priority_glad_module._load_target_dataset_cached.cache_clear()
    monkeypatch.setattr(priority_glad_module, "glad_asset_dir", lambda: asset_dir)


def test_panel_loads_tau_selector_and_target_columns(tmp_path: Path, qapp, monkeypatch: pytest.MonkeyPatch) -> None:
    asset_dir = _write_asset_dir(tmp_path / "assets")
    priority_csv = _write_priority_csv(tmp_path / "priority.csv")
    _patch_assets(monkeypatch, asset_dir)
    panel = PriorityGladPanel()
    panel.priority_csv_edit.setText(str(priority_csv))
    panel.load_matches()
    QApplication.processEvents()

    assert panel.audit_tau_combo.count() == 2
    assert panel.audit_tau_combo.itemText(0) == "0.95"
    assert panel.table.columnCount() == 11
    assert panel.table.horizontalHeaderItem(4).text() == "ecoinventEFv3.7"
    assert panel.table.horizontalHeaderItem(7).text() == "IDEA_EFv2.3"
    assert panel.table.rowCount() == 2


def test_unmatched_filter_reduces_table_and_detail_shows_candidates(tmp_path: Path, qapp, monkeypatch: pytest.MonkeyPatch) -> None:
    asset_dir = _write_asset_dir(tmp_path / "assets")
    priority_csv = _write_priority_csv(tmp_path / "priority.csv")
    _patch_assets(monkeypatch, asset_dir)
    panel = PriorityGladPanel()
    panel.priority_csv_edit.setText(str(priority_csv))
    panel.load_matches()
    panel.target_filter_combo.setCurrentIndex(panel.target_filter_combo.findData("ecoinventEFv3.7"))
    panel.status_filter_combo.setCurrentIndex(panel.status_filter_combo.findData("unmatched"))
    QApplication.processEvents()

    assert panel.table.rowCount() == 1
    assert panel.table.item(0, 1).text() == "flow-b"
    assert '"candidate_count": 2' in panel.detail_box.toPlainText()


def test_export_actions_write_expected_files(tmp_path: Path, qapp, monkeypatch: pytest.MonkeyPatch) -> None:
    asset_dir = _write_asset_dir(tmp_path / "assets")
    priority_csv = _write_priority_csv(tmp_path / "priority.csv")
    output_dir = tmp_path / "out"
    filtered_csv = tmp_path / "filtered.csv"
    unmatched_csv = tmp_path / "unmatched.csv"
    _patch_assets(monkeypatch, asset_dir)
    panel = PriorityGladPanel()
    panel.priority_csv_edit.setText(str(priority_csv))
    panel.output_dir_edit.setText(str(output_dir))
    panel.load_matches()
    panel.target_filter_combo.setCurrentIndex(panel.target_filter_combo.findData("ecoinventEFv3.7"))
    QApplication.processEvents()

    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *args, **kwargs: QMessageBox.Ok))
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *args, **kwargs: QMessageBox.Ok))
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *args, **kwargs: (str(filtered_csv), "CSV files (*.csv)")))
    panel.export_filtered_table()
    assert filtered_csv.exists()

    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *args, **kwargs: (str(unmatched_csv), "CSV files (*.csv)")))
    panel.status_filter_combo.setCurrentIndex(panel.status_filter_combo.findData("unmatched"))
    QApplication.processEvents()
    panel.export_unmatched_for_target()
    assert unmatched_csv.exists()

    panel.export_all_reports()
    assert (output_dir / "priority_glad_summary.json").exists()
    assert (output_dir / "ecoinventEFv3.7_matches.csv").exists()
