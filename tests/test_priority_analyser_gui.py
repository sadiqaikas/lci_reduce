import csv
import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from lci_reduce.priority_analyser_gui import PriorityAnalyserPanel


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
        "cf_status",
    ]
    rows = [
        {
            "flow_id": "flow-a",
            "flow_name": "Alpha",
            "compartment": "air",
            "subcompartment": "urban air",
            "reference_unit": "kg",
            "occurrence_count": "5",
            "characterised_occurrence_count": "5",
            "tau_entry_min": "0",
            "tau_entry_median": "0.1",
            "tau_entry_max": "0.2",
            "eta_0_95": "0",
            "loss_max_0_95": "0",
            "eta_0_99": "0.99",
            "loss_max_0_99": "0.4",
            "cf_status": "characterised",
        },
        {
            "flow_id": "flow-b",
            "flow_name": "Beta",
            "compartment": "water",
            "subcompartment": "fresh water",
            "reference_unit": "kg",
            "occurrence_count": "3",
            "characterised_occurrence_count": "3",
            "tau_entry_min": "0.5",
            "tau_entry_median": "0.6",
            "tau_entry_max": "0.7",
            "eta_0_95": "0.95",
            "loss_max_0_95": "1",
            "eta_0_99": "0",
            "loss_max_0_99": "0",
            "cf_status": "partly_characterised",
        },
        {
            "flow_id": "flow-c",
            "flow_name": "Gamma",
            "compartment": "",
            "subcompartment": "",
            "reference_unit": "kg",
            "occurrence_count": "1",
            "characterised_occurrence_count": "0",
            "tau_entry_min": "",
            "tau_entry_median": "",
            "tau_entry_max": "",
            "eta_0_95": "0.1",
            "loss_max_0_95": "0.3",
            "eta_0_99": "0.4",
            "loss_max_0_99": "0.7",
            "cf_status": "uncharacterised",
        },
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_distributions_tab_exists(qapp) -> None:
    panel = PriorityAnalyserPanel()
    tab_labels = [panel.tabs.tabText(index) for index in range(panel.tabs.count())]
    assert "Distributions" in tab_labels
    assert panel.distributions_grid.layout().count() == 2
    assert panel.distribution_eta_chart.minimumHeight() >= 240
    assert panel.distribution_bins_label.text() == "10 bins shared across both histograms."
    assert panel.distribution_wider_bins_button.text() == "Wider bins"
    assert panel.distribution_narrower_bins_button.text() == "Narrower bins"


def test_distributions_tab_empty_state_clears_figures(qapp) -> None:
    panel = PriorityAnalyserPanel()
    assert panel.distributions_empty_label.text() == "Load a priority CSV to view distributions."
    assert panel.distributions_grid.isHidden()
    assert panel.distribution_eta_chart.axes.texts[0].get_text() == "No data"
    assert panel.distribution_loss_chart.axes.texts[0].get_text() == "No data"
    assert not panel.distribution_wider_bins_button.isEnabled()
    assert not panel.distribution_narrower_bins_button.isEnabled()


def test_loading_dataset_populates_distribution_summaries(tmp_path: Path, qapp) -> None:
    csv_path = _write_priority_csv(tmp_path / "priority.csv")
    panel = PriorityAnalyserPanel()
    panel.priority_csv_edit.setText(str(csv_path))
    panel.load_priority_file()
    QApplication.processEvents()

    assert not panel.distributions_grid.isHidden()
    assert panel.distribution_eta_summary_label.text() == "Rows: 3 | zero: 1 | positive: 2 | eta = 0.95: 1"
    assert panel.distribution_loss_summary_label.text() == "Rows: 3 | zero: 1 | positive: 2 | loss_max = 1: 1"
    assert panel.distribution_eta_chart.axes.get_title() == "eta distribution (0.95)"
    assert panel.distribution_loss_chart.axes.get_title() == "loss_max distribution (0.95)"
    assert len(panel.distribution_eta_chart.axes.patches) > 0
    assert len(panel.distribution_loss_chart.axes.patches) > 0


def test_changing_audit_tau_refreshes_distribution_summaries(tmp_path: Path, qapp) -> None:
    csv_path = _write_priority_csv(tmp_path / "priority.csv")
    panel = PriorityAnalyserPanel()
    panel.priority_csv_edit.setText(str(csv_path))
    panel.load_priority_file()

    summary_095 = panel.distribution_eta_summary_label.text()
    loss_095 = panel.distribution_loss_summary_label.text()

    index_099 = panel.audit_tau_combo.findData("0_99")
    assert index_099 >= 0
    panel.audit_tau_combo.setCurrentIndex(index_099)
    QApplication.processEvents()

    assert panel.distribution_eta_summary_label.text() != summary_095
    assert panel.distribution_eta_summary_label.text() == "Rows: 3 | zero: 1 | positive: 2 | eta = 0.99: 1"
    assert panel.distribution_loss_summary_label.text() != loss_095
    assert panel.distribution_loss_summary_label.text() == "Rows: 3 | zero: 1 | positive: 2 | loss_max = 1: 0"
    assert panel.distribution_eta_chart.axes.get_title() == "eta distribution (0.99)"
    assert panel.distribution_loss_chart.axes.get_title() == "loss_max distribution (0.99)"


def test_distribution_bin_controls_adjust_histogram_resolution(tmp_path: Path, qapp) -> None:
    csv_path = _write_priority_csv(tmp_path / "priority.csv")
    panel = PriorityAnalyserPanel()
    panel.priority_csv_edit.setText(str(csv_path))
    panel.load_priority_file()
    QApplication.processEvents()

    default_eta_patch_count = len(panel.distribution_eta_chart.axes.patches)
    default_loss_patch_count = len(panel.distribution_loss_chart.axes.patches)
    assert panel.distribution_bins_label.text() == "10 bins shared across both histograms."
    assert panel.distribution_wider_bins_button.isEnabled()
    assert panel.distribution_narrower_bins_button.isEnabled()

    panel.distribution_wider_bins_button.click()
    QApplication.processEvents()

    assert panel.distribution_bins_label.text() == "9 bins shared across both histograms."
    assert len(panel.distribution_eta_chart.axes.patches) == default_eta_patch_count - 1
    assert len(panel.distribution_loss_chart.axes.patches) == default_loss_patch_count - 1

    panel.distribution_narrower_bins_button.click()
    QApplication.processEvents()

    assert panel.distribution_bins_label.text() == "10 bins shared across both histograms."
    assert len(panel.distribution_eta_chart.axes.patches) == default_eta_patch_count
    assert len(panel.distribution_loss_chart.axes.patches) == default_loss_patch_count


def test_core_tail_flow_name_column_has_scrollable_width(qapp) -> None:
    panel = PriorityAnalyserPanel()
    assert panel.core_table.columnWidth(1) >= 360
    assert panel.tail_table.columnWidth(1) >= 360
