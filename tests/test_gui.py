import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mplconfig")

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from lci_reduce.gui import MainWindow


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_method_selection_fields_expose_help_text(qapp) -> None:
    window = MainWindow()
    assert window.selection_edit.text() == "all"
    assert window.priority_selection_edit.text() == "all"
    assert window.ecospold_selection_edit.text() == "all"
    assert window.ecospold_priority_selection_edit.text() == "all"
    assert window.diagnostic_selection_edit.text() == "all"
    assert window.selection_help_button.text() == "?"
    assert window.priority_selection_help_button.text() == "?"
    assert window.ecospold_selection_help_button.text() == "?"
    assert window.ecospold_priority_selection_help_button.text() == "?"
    assert window.diagnostic_selection_help_button.text() == "?"
    assert "family:<text>" in window.selection_edit.toolTip()
    assert "method:IPCC 2021" in window.priority_selection_edit.toolTip()
    assert "family:<text>" in window.ecospold_selection_edit.toolTip()
    assert "method:IPCC 2021" in window.ecospold_priority_selection_edit.toolTip()
    assert "family:<text>" in window.diagnostic_selection_edit.toolTip()


def test_cli_info_commands_track_entered_paths(qapp) -> None:
    window = MainWindow()
    window.database_edit.setText("/tmp/source db.zip")
    window.methods_edit.setText("/tmp/methods set.zip")
    window.output_edit.setText("/tmp/out dir")

    window._sync_cli_info_fields_from_forms()
    window.cli_profile_name_edit.setText("demo")
    window.cli_output_path_edit.setText("/tmp/out dir")
    window.cli_priority_csv_path_edit.setText("/tmp/out dir/lcia_flow_priority.csv")
    window.cli_metadata_json_path_edit.setText("/tmp/out dir/lcia_flow_priority_metadata.json")
    window._refresh_cli_commands()

    inspect_command = window.cli_command_boxes["1. Inspect Before You Run"].toPlainText()
    create_command = window.cli_command_boxes["2. Create A Lite Database"].toPlainText()
    analyse_command = window.cli_command_boxes["5. Analyse An Existing Priority File"].toPlainText()

    assert "# Inspect demo before any write operation" in inspect_command
    assert "--database '/tmp/source db.zip'" in inspect_command
    assert "--methods '/tmp/methods set.zip'" in inspect_command
    assert "--output '/tmp/out dir'" in create_command
    assert "--priority-csv" in analyse_command
    assert "--metadata-json" in analyse_command


def test_cli_info_copy_all_commands_updates_clipboard(qapp) -> None:
    window = MainWindow()
    window.copy_all_cli_commands()

    clipboard_text = QApplication.clipboard().text()
    assert "1. Inspect Before You Run" in clipboard_text
    assert "lci_reduce create" in clipboard_text


def test_greedy_exact_tab_defaults(qapp) -> None:
    window = MainWindow()
    assert window.diagnostic_sign_mode_combo.currentData() == "both"
    assert window.diagnostic_tau_edit.text() == "0.95"
    assert window.diagnostic_process_edit.placeholderText() == "Exact process name or UUID"
    assert window.allow_water_mass_volume_override.isChecked() is False
    assert window.priority_allow_water_mass_volume_override.isChecked() is False
    assert window.ecospold_allow_water_mass_volume_override.isChecked() is False
    assert window.ecospold_priority_allow_water_mass_volume_override.isChecked() is False
    assert window.diagnostic_allow_water_mass_volume_override.isChecked() is False


def test_ambiguity_tab_defaults(qapp) -> None:
    window = MainWindow()
    tab_widget = window.centralWidget().layout().itemAt(0).widget()
    tab_labels = [tab_widget.tabText(index) for index in range(tab_widget.count())]

    assert "Explore ambiguities (JSON-LD)" not in tab_labels
    assert window.ambiguity_selection_edit.text() == "all"
    assert window.ambiguity_tolerance_edit.text() == "1e-12"
    assert window.ambiguity_strict_units.isChecked() is True
    assert window.ambiguity_allow_water_mass_volume_override.isChecked() is False
    assert window.ambiguity_button is not None


def test_ecospold_ambiguity_tab_defaults(qapp) -> None:
    window = MainWindow()
    tab_widget = window.centralWidget().layout().itemAt(0).widget()
    tab_labels = [tab_widget.tabText(index) for index in range(tab_widget.count())]

    assert "Explore ambiguities (EcoSpold1)" not in tab_labels
    assert window.ecospold_ambiguity_selection_edit.text() == "all"
    assert window.ecospold_ambiguity_tolerance_edit.text() == "1e-12"
    assert window.ecospold_ambiguity_strict_units.isChecked() is True
    assert window.ecospold_ambiguity_allow_water_mass_volume_override.isChecked() is False
    assert window.ecospold_ambiguity_button is not None


def test_ecospold_tabs_exist_with_expected_defaults(qapp) -> None:
    window = MainWindow()
    tab_widget = window.centralWidget().layout().itemAt(0).widget()
    tab_labels = [tab_widget.tabText(index) for index in range(tab_widget.count())]

    assert "EcoSpold reducer" not in tab_labels
    assert "EcoSpold priority" not in tab_labels
    assert "Greedy vs exact" not in tab_labels
    assert "Priority -> GLAD" in tab_labels
    assert window.ecospold_tau_edit.text() == "0.95"
    assert window.ecospold_priority_audit_tau_edit.text() == "0.95, 0.99"
