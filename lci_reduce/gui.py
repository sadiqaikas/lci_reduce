"""PySide6 GUI for lci_reduce."""

from __future__ import annotations

import csv
import json
import shlex
import sys
import threading
import uuid
from pathlib import Path

from PySide6.QtCore import QEasingCurve, QObject, QPropertyAnimation, QThread, Qt, Signal, Slot
from PySide6.QtGui import QBrush, QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

try:  # pragma: no cover - depends on local Qt build
    from PySide6.QtCharts import QChart, QChartView, QLineSeries, QValueAxis

    QT_CHARTS_AVAILABLE = True
except Exception:  # pragma: no cover - fallback path
    QChart = QChartView = QLineSeries = QValueAxis = None
    QT_CHARTS_AVAILABLE = False

from .cli import CLI_GUIDE_SECTIONS, create_command, inspect_command, priority_command
from .errors import RunCancelledError
from .ambiguity_explorer import explore_ambiguities, AmbiguityExploreConfig
from .greedy_exact_diagnostic import (
    diagnostic_warning_text,
    normalise_diagnostic_tau_values,
    run_greedy_exact_diagnostic,
    GreedyExactDiagnosticConfig,
)
from .models import CreateProgressUpdate, DatabaseReductionGroup, TauReductionRun
from .reduction_curves import clone_run, curve_point_is_valid, export_curve_rows, extract_run_metadata, group_warnings


def _format_int(value: int | None) -> str:
    return "-" if value is None else f"{value:,}"


def _format_percent(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def _sort_runs(runs: list[TauReductionRun]) -> list[TauReductionRun]:
    def key(run: TauReductionRun) -> tuple[float, str]:
        tau = run.tau if run.tau is not None else float("inf")
        return (tau, run.sourceFileName.lower())

    return sorted(runs, key=key)


class InspectWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)
    done = Signal()

    def __init__(self, database: str, methods: str | None) -> None:
        super().__init__()
        self.database = database
        self.methods = methods

    @Slot()
    def run(self) -> None:
        try:
            self.finished.emit(inspect_command(self.database, self.methods))
        except Exception as exc:  # pragma: no cover - GUI guard
            self.failed.emit(str(exc))
        finally:
            self.done.emit()


class CreateWorker(QObject):
    progress = Signal(object)
    finished = Signal(object)
    failed = Signal(str)
    done = Signal()

    def __init__(
        self,
        *,
        database: str,
        methods: str | None,
        output: str,
        tau: float,
        method_selection: str,
        uncharacterised_policy: str,
        strict_units: bool,
        tolerance: float,
        allow_water_mass_volume_override: bool,
    ) -> None:
        super().__init__()
        self.database = database
        self.methods = methods
        self.output = output
        self.tau = tau
        self.method_selection = method_selection
        self.uncharacterised_policy = uncharacterised_policy
        self.strict_units = strict_units
        self.tolerance = tolerance
        self.allow_water_mass_volume_override = allow_water_mass_volume_override

    @Slot()
    def run(self) -> None:
        try:
            result = create_command(
                database=self.database,
                methods=self.methods,
                output=self.output,
                tau=self.tau,
                method_selection=self.method_selection,
                uncharacterised_policy=self.uncharacterised_policy,
                strict_units=self.strict_units,
                tolerance=self.tolerance,
                allow_water_mass_volume_override=self.allow_water_mass_volume_override,
                progress_callback=self.progress.emit,
            )
            self.finished.emit(result)
        except Exception as exc:  # pragma: no cover - GUI guard
            self.failed.emit(str(exc))
        finally:
            self.done.emit()


class PriorityWorker(QObject):
    progress = Signal(object)
    finished = Signal(object)
    failed = Signal(str)
    done = Signal()

    def __init__(
        self,
        *,
        database: str,
        methods: str | None,
        output: str,
        method_selection: str,
        audit_tau: list[float],
        strict_units: bool,
        tolerance: float,
        allow_water_mass_volume_override: bool,
    ) -> None:
        super().__init__()
        self.database = database
        self.methods = methods
        self.output = output
        self.method_selection = method_selection
        self.audit_tau = audit_tau
        self.strict_units = strict_units
        self.tolerance = tolerance
        self.allow_water_mass_volume_override = allow_water_mass_volume_override

    @Slot()
    def run(self) -> None:
        try:
            result = priority_command(
                database=self.database,
                methods=self.methods,
                output=self.output,
                method_selection=self.method_selection,
                audit_tau=self.audit_tau,
                strict_units=self.strict_units,
                tolerance=self.tolerance,
                allow_water_mass_volume_override=self.allow_water_mass_volume_override,
                progress_callback=self.progress.emit,
            )
            self.finished.emit(result)
        except Exception as exc:  # pragma: no cover - GUI guard
            self.failed.emit(str(exc))
        finally:
            self.done.emit()


class AmbiguityExplorerWorker(QObject):
    progress = Signal(object)
    finished = Signal(object)
    failed = Signal(str)
    done = Signal()

    def __init__(self, config: AmbiguityExploreConfig) -> None:
        super().__init__()
        self.config = config

    @Slot()
    def run(self) -> None:
        try:
            result = explore_ambiguities(
                self.config,
                progress_callback=self.progress.emit,
            )
            self.finished.emit(result)
        except Exception as exc:  # pragma: no cover - GUI guard
            self.failed.emit(str(exc))
        finally:
            self.done.emit()


class GreedyExactDiagnosticWorker(QObject):
    progress = Signal(str, int, int)
    finished = Signal(object)
    failed = Signal(str)
    done = Signal()

    def __init__(self, config: GreedyExactDiagnosticConfig) -> None:
        super().__init__()
        self.config = config

    @Slot()
    def run(self) -> None:
        try:
            result = run_greedy_exact_diagnostic(
                self.config,
                progress_callback=self.progress.emit,
            )
            self.finished.emit(result)
        except Exception as exc:  # pragma: no cover - GUI guard
            self.failed.emit(str(exc))
        finally:
            self.done.emit()


class CurveMetadataWorker(QObject):
    progress = Signal(str, int, int)
    finished = Signal(str, object)
    failed = Signal(str, str)
    cancelled = Signal(str)
    done = Signal()

    def __init__(self, run_id: str, source_path: str) -> None:
        super().__init__()
        self.run_id = run_id
        self.source_path = source_path
        self._cancel_event = threading.Event()

    def cancel(self) -> None:
        self._cancel_event.set()

    @Slot()
    def run(self) -> None:
        try:
            result = extract_run_metadata(
                self.source_path,
                progress_callback=self.progress.emit,
                cancel_callback=self._cancel_event.is_set,
            )
            if self._cancel_event.is_set():
                self.cancelled.emit(self.run_id)
            else:
                self.finished.emit(self.run_id, result)
        except RunCancelledError:
            self.cancelled.emit(self.run_id)
        except Exception as exc:  # pragma: no cover - GUI guard
            self.failed.emit(self.run_id, str(exc))
        finally:
            self.done.emit()


class AnimatedCard(QFrame):
    def __init__(self, title: str, summary: str, content: QWidget, *, expanded: bool = False) -> None:
        super().__init__()
        self.setObjectName("guideCard")
        self._title = title
        self._content = content

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(12)

        self._toggle_button = QPushButton()
        self._toggle_button.setObjectName("cardToggle")
        self._toggle_button.setCheckable(True)
        self._toggle_button.setChecked(expanded)
        self._toggle_button.clicked.connect(self._toggle)
        layout.addWidget(self._toggle_button)

        summary_label = QLabel(summary)
        summary_label.setObjectName("muted")
        summary_label.setWordWrap(True)
        layout.addWidget(summary_label)

        self._content.setMaximumHeight(self._content.sizeHint().height() if expanded else 0)
        layout.addWidget(self._content)

        self._animation = QPropertyAnimation(self._content, b"maximumHeight", self)
        self._animation.setDuration(220)
        self._animation.setEasingCurve(QEasingCurve.OutCubic)
        self._update_toggle_text()

    def _toggle(self, checked: bool) -> None:
        self._animation.stop()
        self._animation.setStartValue(self._content.maximumHeight())
        self._animation.setEndValue(self._content.sizeHint().height() if checked else 0)
        self._animation.start()
        self._update_toggle_text()

    def _update_toggle_text(self) -> None:
        prefix = "▾" if self._toggle_button.isChecked() else "▸"
        self._toggle_button.setText(f"{prefix}  {self._title}")


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("lci_reduce")

        self.database_edit = QLineEdit()
        self.methods_edit = QLineEdit()
        self.output_edit = QLineEdit()
        self.tau_edit = QLineEdit("0.95")
        self.selection_edit = QLineEdit("all")
        self.policy_combo = QComboBox()
        self.policy_combo.addItems(["keep", "drop", "fail"])
        self.policy_combo.setCurrentText("drop")
        self.strict_units = QCheckBox("Enforce strict unit compatibility")
        self.strict_units.setChecked(True)
        self.allow_water_mass_volume_override = QCheckBox(
            "Allow water mass/volume override (1 kg = 0.001 m3)"
        )
        self.priority_database_edit = QLineEdit()
        self.priority_methods_edit = QLineEdit()
        self.priority_output_edit = QLineEdit()
        self.priority_selection_edit = QLineEdit("all")
        self.priority_audit_tau_edit = QLineEdit("0.95, 0.99")
        self.priority_strict_units = QCheckBox("Enforce strict unit compatibility")
        self.priority_strict_units.setChecked(True)
        self.priority_allow_water_mass_volume_override = QCheckBox(
            "Allow water mass/volume override (1 kg = 0.001 m3)"
        )
        self.ambiguity_database_edit = QLineEdit()
        self.ambiguity_methods_edit = QLineEdit()
        self.ambiguity_output_edit = QLineEdit()
        self.ambiguity_selection_edit = QLineEdit("all")
        self.ambiguity_strict_units = QCheckBox("Enforce strict unit compatibility")
        self.ambiguity_strict_units.setChecked(True)
        self.ambiguity_allow_water_mass_volume_override = QCheckBox(
            "Allow water mass/volume override (1 kg = 0.001 m3)"
        )
        self.ambiguity_tolerance_edit = QLineEdit("1e-12")
        self.ecospold_ambiguity_database_edit = QLineEdit()
        self.ecospold_ambiguity_methods_edit = QLineEdit()
        self.ecospold_ambiguity_output_edit = QLineEdit()
        self.ecospold_ambiguity_selection_edit = QLineEdit("all")
        self.ecospold_ambiguity_strict_units = QCheckBox("Enforce strict unit compatibility")
        self.ecospold_ambiguity_strict_units.setChecked(True)
        self.ecospold_ambiguity_allow_water_mass_volume_override = QCheckBox(
            "Allow water mass/volume override (1 kg = 0.001 m3)"
        )
        self.ecospold_ambiguity_tolerance_edit = QLineEdit("1e-12")
        self.ecospold_database_edit = QLineEdit()
        self.ecospold_methods_edit = QLineEdit()
        self.ecospold_output_edit = QLineEdit()
        self.ecospold_tau_edit = QLineEdit("0.95")
        self.ecospold_selection_edit = QLineEdit("all")
        self.ecospold_policy_combo = QComboBox()
        self.ecospold_policy_combo.addItems(["keep", "drop", "fail"])
        self.ecospold_policy_combo.setCurrentText("drop")
        self.ecospold_strict_units = QCheckBox("Enforce strict unit compatibility")
        self.ecospold_strict_units.setChecked(True)
        self.ecospold_allow_water_mass_volume_override = QCheckBox(
            "Allow water mass/volume override (1 kg = 0.001 m3)"
        )
        self.ecospold_priority_database_edit = QLineEdit()
        self.ecospold_priority_methods_edit = QLineEdit()
        self.ecospold_priority_output_edit = QLineEdit()
        self.ecospold_priority_selection_edit = QLineEdit("all")
        self.ecospold_priority_audit_tau_edit = QLineEdit("0.95, 0.99")
        self.ecospold_priority_strict_units = QCheckBox("Enforce strict unit compatibility")
        self.ecospold_priority_strict_units.setChecked(True)
        self.ecospold_priority_allow_water_mass_volume_override = QCheckBox(
            "Allow water mass/volume override (1 kg = 0.001 m3)"
        )
        method_selection_help = self._method_selection_help_text()
        self.selection_edit.setToolTip(method_selection_help)
        self.selection_edit.setWhatsThis(method_selection_help)
        self.priority_selection_edit.setToolTip(method_selection_help)
        self.priority_selection_edit.setWhatsThis(method_selection_help)
        self.ambiguity_selection_edit.setToolTip(method_selection_help)
        self.ambiguity_selection_edit.setWhatsThis(method_selection_help)
        self.ecospold_ambiguity_selection_edit.setToolTip(method_selection_help)
        self.ecospold_ambiguity_selection_edit.setWhatsThis(method_selection_help)
        self.ecospold_selection_edit.setToolTip(method_selection_help)
        self.ecospold_selection_edit.setWhatsThis(method_selection_help)
        self.ecospold_priority_selection_edit.setToolTip(method_selection_help)
        self.ecospold_priority_selection_edit.setWhatsThis(method_selection_help)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.priority_progress_bar = QProgressBar()
        self.priority_progress_bar.setRange(0, 1)
        self.priority_progress_bar.setValue(0)
        self.priority_progress_bar.setTextVisible(False)
        self.ambiguity_progress_bar = QProgressBar()
        self.ambiguity_progress_bar.setRange(0, 1)
        self.ambiguity_progress_bar.setValue(0)
        self.ambiguity_progress_bar.setTextVisible(False)
        self.ecospold_ambiguity_progress_bar = QProgressBar()
        self.ecospold_ambiguity_progress_bar.setRange(0, 1)
        self.ecospold_ambiguity_progress_bar.setValue(0)
        self.ecospold_ambiguity_progress_bar.setTextVisible(False)
        self.ecospold_progress_bar = QProgressBar()
        self.ecospold_progress_bar.setRange(0, 1)
        self.ecospold_progress_bar.setValue(0)
        self.ecospold_progress_bar.setTextVisible(False)
        self.ecospold_priority_progress_bar = QProgressBar()
        self.ecospold_priority_progress_bar.setRange(0, 1)
        self.ecospold_priority_progress_bar.setValue(0)
        self.ecospold_priority_progress_bar.setTextVisible(False)

        self.database_methods_label = QLabel("Select a database archive to check for embedded LCIA methods.")
        self.database_methods_label.setObjectName("muted")
        self.database_methods_label.setWordWrap(True)
        self.priority_database_methods_label = QLabel("Select a database archive to check for embedded LCIA methods.")
        self.priority_database_methods_label.setObjectName("muted")
        self.priority_database_methods_label.setWordWrap(True)
        self.ambiguity_database_methods_label = QLabel(
            "Select a JSON-LD database archive to scan ambiguity records without reducing the database."
        )
        self.ambiguity_database_methods_label.setObjectName("muted")
        self.ambiguity_database_methods_label.setWordWrap(True)
        self.ecospold_ambiguity_database_methods_label = QLabel(
            "Select an EcoSpold1 process archive or folder to scan ambiguity records without reducing the database."
        )
        self.ecospold_ambiguity_database_methods_label.setObjectName("muted")
        self.ecospold_ambiguity_database_methods_label.setWordWrap(True)
        self.ecospold_database_methods_label = QLabel(
            "Select an EcoSpold1 process archive or folder. External impact-method input is optional."
        )
        self.ecospold_database_methods_label.setObjectName("muted")
        self.ecospold_database_methods_label.setWordWrap(True)
        self.ecospold_priority_database_methods_label = QLabel(
            "Select an EcoSpold1 process archive or folder. External impact-method input is optional."
        )
        self.ecospold_priority_database_methods_label.setObjectName("muted")
        self.ecospold_priority_database_methods_label.setWordWrap(True)
        self.status_box = QPlainTextEdit()
        self.status_box.setReadOnly(True)
        self.output_box = QPlainTextEdit()
        self.output_box.setReadOnly(True)
        self.output_box.setMaximumBlockCount(64)
        self.priority_status_box = QPlainTextEdit()
        self.priority_status_box.setReadOnly(True)
        self.priority_output_box = QPlainTextEdit()
        self.priority_output_box.setReadOnly(True)
        self.priority_output_box.setMaximumBlockCount(32)
        self.ambiguity_status_box = QPlainTextEdit()
        self.ambiguity_status_box.setReadOnly(True)
        self.ambiguity_output_box = QPlainTextEdit()
        self.ambiguity_output_box.setReadOnly(True)
        self.ambiguity_output_box.setMaximumBlockCount(32)
        self.ecospold_ambiguity_status_box = QPlainTextEdit()
        self.ecospold_ambiguity_status_box.setReadOnly(True)
        self.ecospold_ambiguity_output_box = QPlainTextEdit()
        self.ecospold_ambiguity_output_box.setReadOnly(True)
        self.ecospold_ambiguity_output_box.setMaximumBlockCount(32)
        self.ecospold_status_box = QPlainTextEdit()
        self.ecospold_status_box.setReadOnly(True)
        self.ecospold_output_box = QPlainTextEdit()
        self.ecospold_output_box.setReadOnly(True)
        self.ecospold_output_box.setMaximumBlockCount(32)
        self.ecospold_priority_status_box = QPlainTextEdit()
        self.ecospold_priority_status_box.setReadOnly(True)
        self.ecospold_priority_output_box = QPlainTextEdit()
        self.ecospold_priority_output_box.setReadOnly(True)
        self.ecospold_priority_output_box.setMaximumBlockCount(32)

        self.stage_value = QLabel("Idle")
        self.process_value = QLabel("0 / 0")
        self.exchange_value = QLabel("0 / 0")
        self.current_process_value = QLabel("Ready")
        self.current_process_value.setWordWrap(True)
        self.priority_stage_value = QLabel("Idle")
        self.priority_process_value = QLabel("0 / 0")
        self.priority_current_process_value = QLabel("Ready")
        self.priority_current_process_value.setWordWrap(True)
        self.ambiguity_stage_value = QLabel("Idle")
        self.ambiguity_process_value = QLabel("0 / 0")
        self.ambiguity_current_process_value = QLabel("Ready")
        self.ambiguity_current_process_value.setWordWrap(True)
        self.ecospold_ambiguity_stage_value = QLabel("Idle")
        self.ecospold_ambiguity_process_value = QLabel("0 / 0")
        self.ecospold_ambiguity_current_process_value = QLabel("Ready")
        self.ecospold_ambiguity_current_process_value.setWordWrap(True)
        self.ecospold_stage_value = QLabel("Idle")
        self.ecospold_process_value = QLabel("0 / 0")
        self.ecospold_exchange_value = QLabel("0 / 0")
        self.ecospold_current_process_value = QLabel("Ready")
        self.ecospold_current_process_value.setWordWrap(True)
        self.ecospold_priority_stage_value = QLabel("Idle")
        self.ecospold_priority_process_value = QLabel("0 / 0")
        self.ecospold_priority_current_process_value = QLabel("Ready")
        self.ecospold_priority_current_process_value.setWordWrap(True)

        self.inspect_button: QPushButton | None = None
        self.create_button: QPushButton | None = None
        self.priority_button: QPushButton | None = None
        self.ambiguity_button: QPushButton | None = None
        self.ecospold_ambiguity_button: QPushButton | None = None
        self.ecospold_create_button: QPushButton | None = None
        self.ecospold_priority_button: QPushButton | None = None
        self.use_database_methods_button: QPushButton | None = None
        self.priority_use_database_methods_button: QPushButton | None = None
        self.ecospold_ambiguity_use_database_methods_button: QPushButton | None = None
        self.selection_help_button = self._make_help_button(
            "Show method-selection syntax and examples.",
            self.show_method_selection_help,
        )
        self.priority_selection_help_button = self._make_help_button(
            "Show method-selection syntax and examples.",
            self.show_method_selection_help,
        )
        self.ambiguity_selection_help_button = self._make_help_button(
            "Show method-selection syntax and examples.",
            self.show_method_selection_help,
        )
        self.ecospold_selection_help_button = self._make_help_button(
            "Show method-selection syntax and examples.",
            self.show_method_selection_help,
        )
        self.ecospold_ambiguity_selection_help_button = self._make_help_button(
            "Show method-selection syntax and examples.",
            self.show_method_selection_help,
        )
        self.ecospold_priority_selection_help_button = self._make_help_button(
            "Show method-selection syntax and examples.",
            self.show_method_selection_help,
        )
        self.database_has_impact_methods = False
        self.priority_database_has_impact_methods = False
        self.ambiguity_database_has_impact_methods = False
        self.ecospold_ambiguity_database_has_impact_methods = False
        self.diagnostic_database_has_impact_methods = False

        self._reduction_thread: QThread | None = None
        self._reduction_worker: QObject | None = None
        self._reduction_mode = ""

        self.diagnostic_database_edit = QLineEdit()
        self.diagnostic_methods_edit = QLineEdit()
        self.diagnostic_output_edit = QLineEdit()
        self.diagnostic_selection_edit = QLineEdit("all")
        self.diagnostic_tau_edit = QLineEdit("0.95")
        self.diagnostic_process_edit = QLineEdit()
        self.diagnostic_process_edit.setPlaceholderText("Exact process name or UUID")
        self.diagnostic_sign_mode_combo = QComboBox()
        self.diagnostic_sign_mode_combo.addItem("Both signs", "both")
        self.diagnostic_sign_mode_combo.addItem("Positive only", "positive")
        self.diagnostic_sign_mode_combo.addItem("Negative only", "negative")
        self.diagnostic_strict_units = QCheckBox("Enforce strict unit compatibility")
        self.diagnostic_strict_units.setChecked(True)
        self.diagnostic_allow_water_mass_volume_override = QCheckBox(
            "Allow water mass/volume override (1 kg = 0.001 m3)"
        )
        self.diagnostic_selection_help_button = self._make_help_button(
            "Show method-selection syntax and examples.",
            self.show_method_selection_help,
        )
        self.diagnostic_selection_edit.setToolTip(method_selection_help)
        self.diagnostic_selection_edit.setWhatsThis(method_selection_help)
        self.diagnostic_database_methods_label = QLabel("Select a database archive to check for embedded LCIA methods.")
        self.diagnostic_database_methods_label.setObjectName("muted")
        self.diagnostic_database_methods_label.setWordWrap(True)
        self.diagnostic_run_button: QPushButton | None = None
        self.diagnostic_use_database_methods_button: QPushButton | None = None
        self.diagnostic_status_box = QPlainTextEdit()
        self.diagnostic_status_box.setReadOnly(True)
        self.diagnostic_output_box = QPlainTextEdit()
        self.diagnostic_output_box.setReadOnly(True)
        self.diagnostic_output_box.setMaximumBlockCount(32)
        self.diagnostic_progress_bar = QProgressBar()
        self.diagnostic_progress_bar.setRange(0, 1)
        self.diagnostic_progress_bar.setValue(0)
        self.diagnostic_progress_bar.setTextVisible(False)
        self.diagnostic_stage_value = QLabel("Idle")
        self.diagnostic_runtime_value = QLabel("-")
        self.diagnostic_process_value = QLabel("Select a process")
        self.diagnostic_process_value.setWordWrap(True)
        self.diagnostic_protected_value = QLabel("-")
        self.diagnostic_greedy_coverage_value = QLabel("-")
        self.diagnostic_exact_coverage_value = QLabel("-")
        self.diagnostic_certificate_value = QLabel("-")

        self.curve_group_name_edit = QLineEdit()
        self.curve_group_name_edit.setPlaceholderText("Database group name")
        self.curve_groups: list[DatabaseReductionGroup] = []
        self.curve_status_label = QLabel("No reduction-curve uploads in progress.")
        self.curve_status_label.setObjectName("muted")
        self.curve_progress_bar = QProgressBar()
        self.curve_progress_bar.setRange(0, 1)
        self.curve_progress_bar.setValue(0)
        self.curve_progress_bar.setTextVisible(False)
        self.curve_groups_layout = QVBoxLayout()
        self.curve_groups_layout.setContentsMargins(0, 0, 0, 0)
        self.curve_groups_layout.setSpacing(12)
        self.curve_export_button: QPushButton | None = None
        self.curve_removed_chart: QChartView | QLabel | None = None
        self.curve_retained_chart: QChartView | QLabel | None = None

        self._curve_cache: dict[str, TauReductionRun] = {}
        self._curve_thread: QThread | None = None
        self._curve_worker: CurveMetadataWorker | None = None
        self._curve_active_run_id: str | None = None
        self._curve_queue: list[str] = []
        self._curve_removed_run_ids: set[str] = set()
        self._hidden_tabs: list[QWidget] = []
        self._priority_glad_tab_index: int | None = None
        self._priority_glad_loaded = False
        self._priority_analyser_tab_index: int | None = None
        self._priority_analyser_loaded = False
        self._tabs: QTabWidget | None = None

        self.cli_profile_name_edit = QLineEdit("my_database")
        self.cli_database_path_edit = QLineEdit()
        self.cli_methods_path_edit = QLineEdit()
        self.cli_output_path_edit = QLineEdit()
        self.cli_priority_csv_path_edit = QLineEdit()
        self.cli_metadata_json_path_edit = QLineEdit()
        self.cli_command_boxes: dict[str, QPlainTextEdit] = {}
        self.cli_note_labels: dict[str, QLabel] = {}
        self.cli_copy_status_label: QLabel | None = None

        self._build()
        self._apply_style()
        self._sync_cli_info_fields_from_forms()
        self._refresh_cli_commands()
        self._refresh_curve_views()

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QWidget {
                background: #f3f5f7;
                color: #18222d;
                font-family: "Avenir Next", "Helvetica Neue", "Arial";
                font-size: 13px;
            }
            QMainWindow {
                background: #f3f5f7;
            }
            QFrame#panel, QFrame#groupCard, QGroupBox {
                background: #ffffff;
                border: 1px solid #d3d9df;
                border-radius: 10px;
            }
            QFrame#guideCard {
                background: #ffffff;
                border: 1px solid #cfd7df;
                border-radius: 10px;
            }
            QGroupBox {
                font-weight: 600;
                margin-top: 10px;
                padding: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }
            QTabWidget::pane {
                border: 1px solid #d3d9df;
                background: #f8fafb;
                border-radius: 10px;
                top: -1px;
            }
            QTabBar::tab {
                background: #e8edf1;
                color: #334155;
                border: 1px solid #d3d9df;
                border-bottom: none;
                padding: 8px 14px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                margin-right: 4px;
            }
            QTabBar::tab:selected {
                background: #ffffff;
                color: #0f172a;
            }
            QLineEdit, QComboBox, QPlainTextEdit, QListWidget, QTableWidget {
                background: #ffffff;
                border: 1px solid #c6ced6;
                border-radius: 8px;
                padding: 6px;
            }
            QPlainTextEdit, QTableWidget {
                selection-background-color: #d7e5ef;
            }
            QPushButton {
                background: #234b63;
                color: #ffffff;
                border: 0;
                border-radius: 8px;
                padding: 8px 14px;
                font-weight: 600;
            }
            QPushButton:disabled {
                background: #a7b4bf;
                color: #edf2f7;
            }
            QPushButton#secondary {
                background: #e8edf1;
                color: #243341;
            }
            QPushButton#cardToggle {
                background: transparent;
                color: #12202d;
                border: 0;
                padding: 0;
                font-size: 16px;
                font-weight: 700;
                text-align: left;
            }
            QPushButton#danger {
                background: #8c3a2f;
                color: #ffffff;
            }
            QLabel#sectionTitle {
                font-size: 20px;
                font-weight: 700;
            }
            QLabel#muted {
                color: #5e6b78;
            }
            QLabel#statLabel {
                color: #6b7280;
                font-size: 11px;
                font-weight: 700;
                text-transform: uppercase;
            }
            QLabel#statValue {
                font-size: 17px;
                font-weight: 700;
            }
            QLabel#groupTitle {
                font-size: 15px;
                font-weight: 700;
            }
            QLabel#codeText {
                font-family: "SF Mono", "Menlo", "Courier New";
                background: #f3f6f8;
                border: 1px solid #d9e1e7;
                border-radius: 8px;
                padding: 8px;
            }
            QLabel#eyebrow {
                color: #315067;
                font-size: 11px;
                font-weight: 700;
                text-transform: uppercase;
            }
            QLabel#warningText {
                color: #8a5200;
            }
            QProgressBar {
                background: #e6ebef;
                border: 0;
                border-radius: 8px;
                min-height: 14px;
            }
            QProgressBar::chunk {
                background: #2f6f5e;
                border-radius: 8px;
            }
            QHeaderView::section {
                background: #eef2f5;
                color: #334155;
                border: 0;
                border-right: 1px solid #d6dde4;
                border-bottom: 1px solid #d6dde4;
                padding: 6px;
                font-weight: 600;
            }
            QTableWidget {
                gridline-color: #e3e8ee;
            }
            QScrollArea#cliGuideScroll {
                border: 0;
                background: transparent;
            }
            """
        )

    def _picker_row(
        self,
        line_edit: QLineEdit,
        button_text: str,
        callback,
        *,
        extra_buttons: list[QPushButton] | None = None,
    ) -> QWidget:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(line_edit)
        button = QPushButton(button_text)
        button.setObjectName("secondary")
        button.clicked.connect(callback)
        row.addWidget(button)
        for extra_button in extra_buttons or []:
            row.addWidget(extra_button)
        container = QWidget()
        container.setLayout(row)
        return container

    def _make_help_button(self, tooltip: str, callback) -> QPushButton:
        button = QPushButton("?")
        button.setObjectName("secondary")
        button.setFixedWidth(34)
        button.setToolTip(tooltip)
        button.clicked.connect(callback)
        return button

    def _method_selection_row(self, line_edit: QLineEdit, help_button: QPushButton) -> QWidget:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        row.addWidget(line_edit, 1)
        row.addWidget(help_button)
        container = QWidget()
        container.setLayout(row)
        return container

    @staticmethod
    def _method_selection_help_text() -> str:
        return (
            "Accepted method-selection forms:\n"
            "- all\n"
            "- family:<text>\n"
            "- method:<method name or uuid>\n"
            "- category:<category name or uuid>\n\n"
            "Examples:\n"
            "- all\n"
            "- family:ReCiPe\n"
            "- method:IPCC 2021\n"
            "- category:Climate change\n\n"
            "Matching rules:\n"
            "- family:<text> uses text matching.\n"
            "- method:<text> and category:<text> first try an exact uuid, otherwise a name match.\n"
            "- If a name matches multiple methods or categories, use the uuid instead.\n\n"
            "Run Inspect first to list the available method and category names for the selected inputs."
        )

    def show_method_selection_help(self) -> None:
        QMessageBox.information(self, "Method selection syntax", self._method_selection_help_text())

    def _make_stat_card(self, label: str, value_label: QLabel) -> QWidget:
        box = QFrame()
        box.setObjectName("panel")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(12, 10, 12, 10)
        title = QLabel(label)
        title.setObjectName("statLabel")
        value_label.setObjectName("statValue")
        layout.addWidget(title)
        layout.addWidget(value_label)
        return box

    def _make_section_header(self, title_text: str, body_text: str) -> QWidget:
        frame = QFrame()
        frame.setObjectName("panel")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(16, 14, 16, 14)
        title = QLabel(title_text)
        title.setObjectName("sectionTitle")
        body = QLabel(body_text)
        body.setObjectName("muted")
        body.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(body)
        return frame

    def _build(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        tabs = QTabWidget()
        self._tabs = tabs
        tabs.addTab(self._build_reduction_tab(), "Reduction")
        tabs.addTab(self._build_priority_tab(), "Flow priority")
        self._hidden_tabs = [
            self._build_ambiguity_tab(),
            self._build_ecospold_ambiguity_tab(),
            self._build_ecospold_reduction_tab(),
            self._build_ecospold_priority_tab(),
            self._build_greedy_exact_tab(),
        ]
        self._priority_glad_tab_index = tabs.addTab(self._build_priority_glad_placeholder(), "Priority -> GLAD")
        self._priority_analyser_tab_index = tabs.addTab(self._build_priority_analyser_placeholder(), "Priority analyser")
        tabs.addTab(self._build_curves_tab(), "Reduction curves")
        tabs.addTab(self._build_cli_tab(), "CLI info")
        tabs.currentChanged.connect(self._maybe_load_lazy_tabs)
        root.addWidget(tabs)

        self.setCentralWidget(central)

    def _build_priority_analyser_placeholder(self) -> QWidget:
        frame = QFrame()
        frame.setObjectName("panel")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(16, 14, 16, 14)
        title = QLabel("Priority analyser loads on demand.")
        title.setObjectName("sectionTitle")
        body = QLabel(
            "This tab is created lazily to keep GUI startup light. Open the tab to load the analyser and plots."
        )
        body.setWordWrap(True)
        body.setObjectName("muted")
        layout.addWidget(title)
        layout.addWidget(body)
        return frame

    def _build_priority_glad_placeholder(self) -> QWidget:
        frame = QFrame()
        frame.setObjectName("panel")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(16, 14, 16, 14)
        title = QLabel("Priority -> GLAD loads on demand.")
        title.setObjectName("sectionTitle")
        body = QLabel(
            "This tab is created lazily to keep GUI startup light. Open the tab to load direct target matching against bundled GLAD flow-list snapshots."
        )
        body.setWordWrap(True)
        body.setObjectName("muted")
        layout.addWidget(title)
        layout.addWidget(body)
        return frame

    def _maybe_load_lazy_tabs(self, index: int) -> None:
        if self._priority_glad_tab_index is not None and index == self._priority_glad_tab_index:
            self._load_priority_glad_tab()
        if self._priority_analyser_tab_index is not None and index == self._priority_analyser_tab_index:
            self._load_priority_analyser_tab()

    def _load_priority_glad_tab(self) -> None:
        if self._priority_glad_loaded or self._tabs is None or self._priority_glad_tab_index is None:
            return
        from .priority_glad_gui import PriorityGladPanel

        index = self._priority_glad_tab_index
        placeholder = self._tabs.widget(index)
        if placeholder is not None:
            self._tabs.removeTab(index)
            placeholder.deleteLater()
        panel = PriorityGladPanel()
        self._tabs.insertTab(index, panel, "Priority -> GLAD")
        self._tabs.setCurrentIndex(index)
        self._priority_glad_loaded = True

    def _maybe_load_priority_analyser_tab(self, index: int) -> None:
        if self._priority_analyser_loaded:
            return
        if self._priority_analyser_tab_index is None or index != self._priority_analyser_tab_index:
            return
        self._load_priority_analyser_tab()

    def _load_priority_analyser_tab(self) -> None:
        if self._priority_analyser_loaded or self._tabs is None or self._priority_analyser_tab_index is None:
            return
        from .priority_analyser_gui import PriorityAnalyserPanel

        index = self._priority_analyser_tab_index
        placeholder = self._tabs.widget(index)
        if placeholder is not None:
            self._tabs.removeTab(index)
            placeholder.deleteLater()
        panel = PriorityAnalyserPanel()
        self._tabs.insertTab(index, panel, "Priority analyser")
        self._tabs.setCurrentIndex(index)
        self._priority_analyser_loaded = True

    def _build_reduction_tab(self) -> QWidget:
        tab = QWidget()
        root = QVBoxLayout(tab)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(12)

        root.addWidget(
            self._make_section_header(
                "LCI database reduction",
                "Deterministic signed tau-cover on openLCA JSON-LD archives with run-time validation artefacts.",
            )
        )

        inputs_group = QGroupBox("Inputs")
        inputs_form = QFormLayout(inputs_group)
        inputs_form.setSpacing(10)

        self.use_database_methods_button = QPushButton("Use database methods")
        self.use_database_methods_button.setObjectName("secondary")
        self.use_database_methods_button.clicked.connect(self.use_database_methods)
        self.use_database_methods_button.setEnabled(False)

        inputs_form.addRow(
            "Database archive",
            self._picker_row(self.database_edit, "Browse", self.pick_database),
        )
        inputs_form.addRow(
            "Methods archive or folder",
            self._picker_row(
                self.methods_edit,
                "Browse",
                self.pick_methods,
                extra_buttons=[self.use_database_methods_button],
            ),
        )
        inputs_form.addRow("", self.database_methods_label)
        inputs_form.addRow(
            "Output folder",
            self._picker_row(self.output_edit, "Browse", self.pick_output),
        )

        settings_group = QGroupBox("Reduction settings")
        settings_form = QFormLayout(settings_group)
        settings_form.setSpacing(10)
        settings_form.addRow("Tau", self.tau_edit)
        settings_form.addRow("Method selection", self._method_selection_row(self.selection_edit, self.selection_help_button))
        settings_form.addRow("Uncharacterised policy", self.policy_combo)
        settings_form.addRow("", self.strict_units)
        settings_form.addRow("", self.allow_water_mass_volume_override)

        top_row = QHBoxLayout()
        top_row.addWidget(inputs_group, 3)
        top_row.addWidget(settings_group, 2)
        root.addLayout(top_row)

        controls = QHBoxLayout()
        self.inspect_button = QPushButton("Inspect")
        self.inspect_button.setObjectName("secondary")
        self.inspect_button.clicked.connect(self.run_inspect)
        self.create_button = QPushButton("Create reduced database")
        self.create_button.clicked.connect(self.run_create)
        controls.addWidget(self.inspect_button)
        controls.addWidget(self.create_button)
        controls.addStretch(1)
        root.addLayout(controls)

        status_group = QGroupBox("Run status")
        status_layout = QVBoxLayout(status_group)
        stats_grid = QGridLayout()
        stats_grid.addWidget(self._make_stat_card("Stage", self.stage_value), 0, 0)
        stats_grid.addWidget(self._make_stat_card("Processes", self.process_value), 0, 1)
        stats_grid.addWidget(self._make_stat_card("Removed / seen", self.exchange_value), 0, 2)
        stats_grid.addWidget(self._make_stat_card("Current process", self.current_process_value), 1, 0, 1, 3)
        status_layout.addLayout(stats_grid)
        status_layout.addWidget(self.progress_bar)
        root.addWidget(status_group)

        logs_row = QHBoxLayout()
        activity_group = QGroupBox("Run log")
        activity_layout = QVBoxLayout(activity_group)
        activity_layout.addWidget(self.status_box)
        outputs_group = QGroupBox("Output artefacts")
        outputs_layout = QVBoxLayout(outputs_group)
        outputs_layout.addWidget(self.output_box)
        logs_row.addWidget(activity_group, 1)
        logs_row.addWidget(outputs_group, 1)
        root.addLayout(logs_row)
        return tab

    def _build_curves_tab(self) -> QWidget:
        tab = QWidget()
        root = QVBoxLayout(tab)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(12)

        root.addWidget(
            self._make_section_header(
                "Database reduction curves",
                "Compare retained and removed elementary exchanges across tau values using completed reduction runs.",
            )
        )

        controls_frame = QFrame()
        controls_frame.setObjectName("panel")
        controls_layout = QVBoxLayout(controls_frame)
        controls_layout.setContentsMargins(14, 12, 14, 12)
        controls_layout.setSpacing(10)

        add_row = QHBoxLayout()
        add_row.addWidget(self.curve_group_name_edit, 1)
        add_group_button = QPushButton("Add database")
        add_group_button.clicked.connect(self.add_curve_group)
        add_row.addWidget(add_group_button)
        self.curve_export_button = QPushButton("Export CSV")
        self.curve_export_button.setObjectName("secondary")
        self.curve_export_button.clicked.connect(self.export_curve_csv)
        add_row.addWidget(self.curve_export_button)
        controls_layout.addLayout(add_row)
        controls_layout.addWidget(self.curve_status_label)
        controls_layout.addWidget(self.curve_progress_bar)
        root.addWidget(controls_frame)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        groups_container = QWidget()
        groups_container.setLayout(self.curve_groups_layout)
        self.curve_groups_layout.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(groups_container)
        scroll.setMinimumWidth(520)
        splitter.addWidget(scroll)

        charts_panel = QFrame()
        charts_panel.setObjectName("panel")
        charts_layout = QVBoxLayout(charts_panel)
        charts_layout.setContentsMargins(14, 12, 14, 12)
        charts_layout.setSpacing(12)

        charts_title = QLabel("Curves")
        charts_title.setObjectName("sectionTitle")
        charts_layout.addWidget(charts_title)

        if QT_CHARTS_AVAILABLE:
            self.curve_removed_chart = self._make_chart_view()
            self.curve_retained_chart = self._make_chart_view()
            charts_layout.addWidget(self.curve_removed_chart, 1)
            charts_layout.addWidget(self.curve_retained_chart, 1)
        else:  # pragma: no cover - depends on local Qt build
            missing = QLabel("Qt Charts is not available in this environment.")
            missing.setObjectName("muted")
            missing.setWordWrap(True)
            self.curve_removed_chart = missing
            self.curve_retained_chart = QLabel("Chart rendering is unavailable.")
            self.curve_retained_chart.setObjectName("muted")
            charts_layout.addWidget(self.curve_removed_chart)
            charts_layout.addWidget(self.curve_retained_chart)
            charts_layout.addStretch(1)

        splitter.addWidget(charts_panel)
        splitter.setSizes([700, 520])
        root.addWidget(splitter, 1)
        return tab

    def _build_priority_tab(self) -> QWidget:
        tab = QWidget()
        root = QVBoxLayout(tab)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(12)

        root.addWidget(
            self._make_section_header(
                "LCIA flow priority",
                "Generate LCIA transfer-priority sidecars without rewriting the database.",
            )
        )

        inputs_group = QGroupBox("Inputs")
        inputs_form = QFormLayout(inputs_group)
        inputs_form.setSpacing(10)
        self.priority_use_database_methods_button = QPushButton("Use database methods")
        self.priority_use_database_methods_button.setObjectName("secondary")
        self.priority_use_database_methods_button.clicked.connect(self.use_priority_database_methods)
        self.priority_use_database_methods_button.setEnabled(False)
        inputs_form.addRow(
            "Database archive",
            self._picker_row(self.priority_database_edit, "Browse", self.pick_priority_database),
        )
        inputs_form.addRow(
            "Methods archive or folder",
            self._picker_row(
                self.priority_methods_edit,
                "Browse",
                self.pick_priority_methods,
                extra_buttons=[self.priority_use_database_methods_button],
            ),
        )
        inputs_form.addRow("", self.priority_database_methods_label)
        inputs_form.addRow(
            "Output folder",
            self._picker_row(self.priority_output_edit, "Browse", self.pick_priority_output),
        )

        settings_group = QGroupBox("Audit settings")
        settings_form = QFormLayout(settings_group)
        settings_form.setSpacing(10)
        settings_form.addRow(
            "Method selection",
            self._method_selection_row(self.priority_selection_edit, self.priority_selection_help_button),
        )
        settings_form.addRow("Audit tau values", self.priority_audit_tau_edit)
        settings_form.addRow("", self.priority_strict_units)
        settings_form.addRow("", self.priority_allow_water_mass_volume_override)

        top_row = QHBoxLayout()
        top_row.addWidget(inputs_group, 3)
        top_row.addWidget(settings_group, 2)
        root.addLayout(top_row)

        controls = QHBoxLayout()
        self.priority_button = QPushButton("Generate flow priority")
        self.priority_button.clicked.connect(self.run_priority)
        controls.addWidget(self.priority_button)
        controls.addStretch(1)
        root.addLayout(controls)

        status_group = QGroupBox("Run status")
        status_layout = QVBoxLayout(status_group)
        stats_grid = QGridLayout()
        stats_grid.addWidget(self._make_stat_card("Stage", self.priority_stage_value), 0, 0)
        stats_grid.addWidget(self._make_stat_card("Processes", self.priority_process_value), 0, 1)
        stats_grid.addWidget(self._make_stat_card("Current process", self.priority_current_process_value), 1, 0, 1, 2)
        status_layout.addLayout(stats_grid)
        status_layout.addWidget(self.priority_progress_bar)
        root.addWidget(status_group)

        logs_row = QHBoxLayout()
        activity_group = QGroupBox("Run log")
        activity_layout = QVBoxLayout(activity_group)
        activity_layout.addWidget(self.priority_status_box)
        outputs_group = QGroupBox("Output artefacts")
        outputs_layout = QVBoxLayout(outputs_group)
        outputs_layout.addWidget(self.priority_output_box)
        logs_row.addWidget(activity_group, 1)
        logs_row.addWidget(outputs_group, 1)
        root.addLayout(logs_row)
        return tab

    def _build_ambiguity_tab(self) -> QWidget:
        tab = QWidget()
        root = QVBoxLayout(tab)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(12)

        root.addWidget(
            self._make_section_header(
                "Explore LCIA ambiguities from JSON-LD",
                "Scan ambiguity records from a JSON-LD database without reducing it. This is the dedicated review workflow for non-location ambiguity.",
            )
        )

        inputs_group = QGroupBox("Inputs")
        inputs_form = QFormLayout(inputs_group)
        inputs_form.setSpacing(10)
        self.ambiguity_use_database_methods_button = QPushButton("Use database methods")
        self.ambiguity_use_database_methods_button.setObjectName("secondary")
        self.ambiguity_use_database_methods_button.clicked.connect(self.use_ambiguity_database_methods)
        self.ambiguity_use_database_methods_button.setEnabled(False)
        inputs_form.addRow(
            "Database archive",
            self._picker_row(self.ambiguity_database_edit, "Browse", self.pick_ambiguity_database),
        )
        inputs_form.addRow(
            "Methods archive or folder",
            self._picker_row(
                self.ambiguity_methods_edit,
                "Browse",
                self.pick_ambiguity_methods,
                extra_buttons=[self.ambiguity_use_database_methods_button],
            ),
        )
        inputs_form.addRow("", self.ambiguity_database_methods_label)
        inputs_form.addRow(
            "Output folder",
            self._picker_row(self.ambiguity_output_edit, "Browse", self.pick_ambiguity_output),
        )

        settings_group = QGroupBox("Scan settings")
        settings_form = QFormLayout(settings_group)
        settings_form.setSpacing(10)
        settings_form.addRow(
            "Method selection",
            self._method_selection_row(self.ambiguity_selection_edit, self.ambiguity_selection_help_button),
        )
        settings_form.addRow("Tolerance", self.ambiguity_tolerance_edit)
        settings_form.addRow("", self.ambiguity_strict_units)
        settings_form.addRow("", self.ambiguity_allow_water_mass_volume_override)

        top_row = QHBoxLayout()
        top_row.addWidget(inputs_group, 3)
        top_row.addWidget(settings_group, 2)
        root.addLayout(top_row)

        controls = QHBoxLayout()
        self.ambiguity_button = QPushButton("Explore ambiguities")
        self.ambiguity_button.clicked.connect(self.run_ambiguity_explorer)
        controls.addWidget(self.ambiguity_button)
        controls.addStretch(1)
        root.addLayout(controls)

        status_group = QGroupBox("Run status")
        status_layout = QVBoxLayout(status_group)
        stats_grid = QGridLayout()
        stats_grid.addWidget(self._make_stat_card("Stage", self.ambiguity_stage_value), 0, 0)
        stats_grid.addWidget(self._make_stat_card("Processes", self.ambiguity_process_value), 0, 1)
        stats_grid.addWidget(self._make_stat_card("Current process", self.ambiguity_current_process_value), 1, 0, 1, 2)
        status_layout.addLayout(stats_grid)
        status_layout.addWidget(self.ambiguity_progress_bar)
        root.addWidget(status_group)

        logs_row = QHBoxLayout()
        activity_group = QGroupBox("Run log")
        activity_layout = QVBoxLayout(activity_group)
        activity_layout.addWidget(self.ambiguity_status_box)
        outputs_group = QGroupBox("Output artefacts")
        outputs_layout = QVBoxLayout(outputs_group)
        outputs_layout.addWidget(self.ambiguity_output_box)
        logs_row.addWidget(activity_group, 1)
        logs_row.addWidget(outputs_group, 1)
        root.addLayout(logs_row)
        return tab

    def _build_ecospold_ambiguity_tab(self) -> QWidget:
        tab = QWidget()
        root = QVBoxLayout(tab)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(12)

        root.addWidget(
            self._make_section_header(
                "Explore LCIA ambiguities from EcoSpold1",
                "Scan ambiguity records from an EcoSpold1 process archive or folder without reducing it. This uses the same ambiguity engine as the JSON-LD explorer.",
            )
        )

        inputs_group = QGroupBox("Inputs")
        inputs_form = QFormLayout(inputs_group)
        inputs_form.setSpacing(10)
        self.ecospold_ambiguity_use_database_methods_button = QPushButton("Use database methods")
        self.ecospold_ambiguity_use_database_methods_button.setObjectName("secondary")
        self.ecospold_ambiguity_use_database_methods_button.clicked.connect(self.use_ecospold_ambiguity_database_methods)
        self.ecospold_ambiguity_use_database_methods_button.setEnabled(False)
        inputs_form.addRow(
            "Process archive or folder",
            self._picker_row(self.ecospold_ambiguity_database_edit, "Browse", self.pick_ecospold_ambiguity_database),
        )
        inputs_form.addRow(
            "Methods archive or folder",
            self._picker_row(
                self.ecospold_ambiguity_methods_edit,
                "Browse",
                self.pick_ecospold_ambiguity_methods,
                extra_buttons=[self.ecospold_ambiguity_use_database_methods_button],
            ),
        )
        inputs_form.addRow("", self.ecospold_ambiguity_database_methods_label)
        inputs_form.addRow(
            "Output folder",
            self._picker_row(self.ecospold_ambiguity_output_edit, "Browse", self.pick_ecospold_ambiguity_output),
        )

        settings_group = QGroupBox("Scan settings")
        settings_form = QFormLayout(settings_group)
        settings_form.setSpacing(10)
        settings_form.addRow(
            "Method selection",
            self._method_selection_row(
                self.ecospold_ambiguity_selection_edit,
                self.ecospold_ambiguity_selection_help_button,
            ),
        )
        settings_form.addRow("Tolerance", self.ecospold_ambiguity_tolerance_edit)
        settings_form.addRow("", self.ecospold_ambiguity_strict_units)
        settings_form.addRow("", self.ecospold_ambiguity_allow_water_mass_volume_override)

        top_row = QHBoxLayout()
        top_row.addWidget(inputs_group, 3)
        top_row.addWidget(settings_group, 2)
        root.addLayout(top_row)

        controls = QHBoxLayout()
        self.ecospold_ambiguity_button = QPushButton("Explore ambiguities")
        self.ecospold_ambiguity_button.clicked.connect(self.run_ecospold_ambiguity_explorer)
        controls.addWidget(self.ecospold_ambiguity_button)
        controls.addStretch(1)
        root.addLayout(controls)

        status_group = QGroupBox("Run status")
        status_layout = QVBoxLayout(status_group)
        stats_grid = QGridLayout()
        stats_grid.addWidget(self._make_stat_card("Stage", self.ecospold_ambiguity_stage_value), 0, 0)
        stats_grid.addWidget(self._make_stat_card("Processes", self.ecospold_ambiguity_process_value), 0, 1)
        stats_grid.addWidget(
            self._make_stat_card("Current process", self.ecospold_ambiguity_current_process_value),
            1,
            0,
            1,
            2,
        )
        status_layout.addLayout(stats_grid)
        status_layout.addWidget(self.ecospold_ambiguity_progress_bar)
        root.addWidget(status_group)

        logs_row = QHBoxLayout()
        activity_group = QGroupBox("Run log")
        activity_layout = QVBoxLayout(activity_group)
        activity_layout.addWidget(self.ecospold_ambiguity_status_box)
        outputs_group = QGroupBox("Output artefacts")
        outputs_layout = QVBoxLayout(outputs_group)
        outputs_layout.addWidget(self.ecospold_ambiguity_output_box)
        logs_row.addWidget(activity_group, 1)
        logs_row.addWidget(outputs_group, 1)
        root.addLayout(logs_row)
        return tab

    def _build_ecospold_reduction_tab(self) -> QWidget:
        tab = QWidget()
        root = QVBoxLayout(tab)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(12)

        root.addWidget(
            self._make_section_header(
                "EcoSpold1 reduction",
                "Deterministic signed tau-cover on EcoSpold1 process archives using the same LCIA contribution and coverage rules as the JSON-LD reducer.",
            )
        )

        inputs_group = QGroupBox("Inputs")
        inputs_form = QFormLayout(inputs_group)
        inputs_form.setSpacing(10)
        inputs_form.addRow(
            "Process archive or folder",
            self._picker_row(self.ecospold_database_edit, "Browse", self.pick_ecospold_database),
        )
        inputs_form.addRow(
            "Impact-method archive or folder",
            self._picker_row(self.ecospold_methods_edit, "Browse", self.pick_ecospold_methods),
        )
        inputs_form.addRow("", self.ecospold_database_methods_label)
        inputs_form.addRow(
            "Output folder",
            self._picker_row(self.ecospold_output_edit, "Browse", self.pick_ecospold_output),
        )

        settings_group = QGroupBox("Reduction settings")
        settings_form = QFormLayout(settings_group)
        settings_form.setSpacing(10)
        settings_form.addRow("Tau", self.ecospold_tau_edit)
        settings_form.addRow(
            "Method selection",
            self._method_selection_row(self.ecospold_selection_edit, self.ecospold_selection_help_button),
        )
        settings_form.addRow("Uncharacterised policy", self.ecospold_policy_combo)
        settings_form.addRow("", self.ecospold_strict_units)
        settings_form.addRow("", self.ecospold_allow_water_mass_volume_override)

        top_row = QHBoxLayout()
        top_row.addWidget(inputs_group, 3)
        top_row.addWidget(settings_group, 2)
        root.addLayout(top_row)

        controls = QHBoxLayout()
        self.ecospold_create_button = QPushButton("Create reduced EcoSpold archive")
        self.ecospold_create_button.clicked.connect(self.run_ecospold_create)
        controls.addWidget(self.ecospold_create_button)
        controls.addStretch(1)
        root.addLayout(controls)

        status_group = QGroupBox("Run status")
        status_layout = QVBoxLayout(status_group)
        stats_grid = QGridLayout()
        stats_grid.addWidget(self._make_stat_card("Stage", self.ecospold_stage_value), 0, 0)
        stats_grid.addWidget(self._make_stat_card("Processes", self.ecospold_process_value), 0, 1)
        stats_grid.addWidget(self._make_stat_card("Removed / seen", self.ecospold_exchange_value), 0, 2)
        stats_grid.addWidget(self._make_stat_card("Current process", self.ecospold_current_process_value), 1, 0, 1, 3)
        status_layout.addLayout(stats_grid)
        status_layout.addWidget(self.ecospold_progress_bar)
        root.addWidget(status_group)

        logs_row = QHBoxLayout()
        activity_group = QGroupBox("Run log")
        activity_layout = QVBoxLayout(activity_group)
        activity_layout.addWidget(self.ecospold_status_box)
        outputs_group = QGroupBox("Output artefacts")
        outputs_layout = QVBoxLayout(outputs_group)
        outputs_layout.addWidget(self.ecospold_output_box)
        logs_row.addWidget(activity_group, 1)
        logs_row.addWidget(outputs_group, 1)
        root.addLayout(logs_row)
        return tab

    def _build_ecospold_priority_tab(self) -> QWidget:
        tab = QWidget()
        root = QVBoxLayout(tab)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(12)

        root.addWidget(
            self._make_section_header(
                "EcoSpold1 flow priority",
                "Generate LCIA-critical flow-priority sidecars from EcoSpold1 process archives without rewriting the source files.",
            )
        )

        inputs_group = QGroupBox("Inputs")
        inputs_form = QFormLayout(inputs_group)
        inputs_form.setSpacing(10)
        inputs_form.addRow(
            "Process archive or folder",
            self._picker_row(self.ecospold_priority_database_edit, "Browse", self.pick_ecospold_priority_database),
        )
        inputs_form.addRow(
            "Impact-method archive or folder",
            self._picker_row(self.ecospold_priority_methods_edit, "Browse", self.pick_ecospold_priority_methods),
        )
        inputs_form.addRow("", self.ecospold_priority_database_methods_label)
        inputs_form.addRow(
            "Output folder",
            self._picker_row(self.ecospold_priority_output_edit, "Browse", self.pick_ecospold_priority_output),
        )

        settings_group = QGroupBox("Audit settings")
        settings_form = QFormLayout(settings_group)
        settings_form.setSpacing(10)
        settings_form.addRow(
            "Method selection",
            self._method_selection_row(
                self.ecospold_priority_selection_edit,
                self.ecospold_priority_selection_help_button,
            ),
        )
        settings_form.addRow("Audit tau values", self.ecospold_priority_audit_tau_edit)
        settings_form.addRow("", self.ecospold_priority_strict_units)
        settings_form.addRow("", self.ecospold_priority_allow_water_mass_volume_override)

        top_row = QHBoxLayout()
        top_row.addWidget(inputs_group, 3)
        top_row.addWidget(settings_group, 2)
        root.addLayout(top_row)

        controls = QHBoxLayout()
        self.ecospold_priority_button = QPushButton("Generate EcoSpold flow priority")
        self.ecospold_priority_button.clicked.connect(self.run_ecospold_priority)
        controls.addWidget(self.ecospold_priority_button)
        controls.addStretch(1)
        root.addLayout(controls)

        status_group = QGroupBox("Run status")
        status_layout = QVBoxLayout(status_group)
        stats_grid = QGridLayout()
        stats_grid.addWidget(self._make_stat_card("Stage", self.ecospold_priority_stage_value), 0, 0)
        stats_grid.addWidget(self._make_stat_card("Processes", self.ecospold_priority_process_value), 0, 1)
        stats_grid.addWidget(
            self._make_stat_card("Current process", self.ecospold_priority_current_process_value),
            1,
            0,
            1,
            2,
        )
        status_layout.addLayout(stats_grid)
        status_layout.addWidget(self.ecospold_priority_progress_bar)
        root.addWidget(status_group)

        logs_row = QHBoxLayout()
        activity_group = QGroupBox("Run log")
        activity_layout = QVBoxLayout(activity_group)
        activity_layout.addWidget(self.ecospold_priority_status_box)
        outputs_group = QGroupBox("Output artefacts")
        outputs_layout = QVBoxLayout(outputs_group)
        outputs_layout.addWidget(self.ecospold_priority_output_box)
        logs_row.addWidget(activity_group, 1)
        logs_row.addWidget(outputs_group, 1)
        root.addLayout(logs_row)
        return tab

    def _build_greedy_exact_tab(self) -> QWidget:
        tab = QWidget()
        root = QVBoxLayout(tab)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(12)

        root.addWidget(
            self._make_section_header(
                "Single-process greedy vs exact diagnostic",
                "Compare the implemented greedy tau-cover against an exact single-process MILP, then write an importable JSON-LD diagnostic ZIP with cloned process variants.",
            )
        )

        warning_box = QFrame()
        warning_box.setObjectName("panel")
        warning_layout = QVBoxLayout(warning_box)
        warning_layout.setContentsMargins(16, 14, 16, 14)
        warning_label = QLabel(diagnostic_warning_text())
        warning_label.setObjectName("muted")
        warning_label.setWordWrap(True)
        warning_layout.addWidget(warning_label)
        root.addWidget(warning_box)

        inputs_group = QGroupBox("Inputs")
        inputs_form = QFormLayout(inputs_group)
        inputs_form.setSpacing(10)
        self.diagnostic_use_database_methods_button = QPushButton("Use database methods")
        self.diagnostic_use_database_methods_button.setObjectName("secondary")
        self.diagnostic_use_database_methods_button.clicked.connect(self.use_diagnostic_database_methods)
        self.diagnostic_use_database_methods_button.setEnabled(False)
        inputs_form.addRow(
            "Database archive",
            self._picker_row(
                self.diagnostic_database_edit,
                "Browse",
                self.pick_diagnostic_database,
            ),
        )
        inputs_form.addRow(
            "Methods archive or folder",
            self._picker_row(
                self.diagnostic_methods_edit,
                "Browse",
                self.pick_diagnostic_methods,
                extra_buttons=[self.diagnostic_use_database_methods_button],
            ),
        )
        inputs_form.addRow("", self.diagnostic_database_methods_label)
        inputs_form.addRow(
            "Output folder",
            self._picker_row(self.diagnostic_output_edit, "Browse", self.pick_diagnostic_output),
        )
        inputs_form.addRow("Process", self.diagnostic_process_edit)
        process_help = QLabel(
            "Enter the exact process name or UUID. Leave empty to run the same diagnostic over all processes. "
            "If the name is ambiguous, use the UUID."
        )
        process_help.setObjectName("muted")
        process_help.setWordWrap(True)
        inputs_form.addRow("", process_help)

        settings_group = QGroupBox("Diagnostic settings")
        settings_form = QFormLayout(settings_group)
        settings_form.setSpacing(10)
        settings_form.addRow(
            "LCIA selection",
            self._method_selection_row(self.diagnostic_selection_edit, self.diagnostic_selection_help_button),
        )
        settings_form.addRow("Tau values", self.diagnostic_tau_edit)
        settings_form.addRow("Sign mode", self.diagnostic_sign_mode_combo)
        settings_form.addRow("", self.diagnostic_strict_units)
        settings_form.addRow("", self.diagnostic_allow_water_mass_volume_override)

        top_row = QHBoxLayout()
        top_row.addWidget(inputs_group, 3)
        top_row.addWidget(settings_group, 2)
        root.addLayout(top_row)

        controls = QHBoxLayout()
        self.diagnostic_run_button = QPushButton("Run diagnostic")
        self.diagnostic_run_button.clicked.connect(self.run_greedy_exact_diagnostic)
        controls.addWidget(self.diagnostic_run_button)
        controls.addStretch(1)
        root.addLayout(controls)

        summary_group = QGroupBox("Run summary")
        summary_layout = QVBoxLayout(summary_group)
        stats_grid = QGridLayout()
        stats_grid.addWidget(self._make_stat_card("Stage", self.diagnostic_stage_value), 0, 0)
        stats_grid.addWidget(self._make_stat_card("Runtime", self.diagnostic_runtime_value), 0, 1)
        stats_grid.addWidget(self._make_stat_card("Selected process", self.diagnostic_process_value), 0, 2)
        stats_grid.addWidget(self._make_stat_card("Protected exchanges", self.diagnostic_protected_value), 1, 0)
        stats_grid.addWidget(self._make_stat_card("Tau values", self.diagnostic_greedy_coverage_value), 1, 1)
        stats_grid.addWidget(self._make_stat_card("Exact clones", self.diagnostic_exact_coverage_value), 1, 2)
        stats_grid.addWidget(self._make_stat_card("Tau certificate", self.diagnostic_certificate_value), 2, 0, 1, 3)
        summary_layout.addLayout(stats_grid)
        summary_layout.addWidget(self.diagnostic_progress_bar)
        root.addWidget(summary_group)

        logs_row = QHBoxLayout()
        activity_group = QGroupBox("Run log")
        activity_layout = QVBoxLayout(activity_group)
        activity_layout.addWidget(self.diagnostic_status_box)
        outputs_group = QGroupBox("Output artefacts")
        outputs_layout = QVBoxLayout(outputs_group)
        outputs_layout.addWidget(self.diagnostic_output_box)
        logs_row.addWidget(activity_group, 1)
        logs_row.addWidget(outputs_group, 1)
        root.addLayout(logs_row)
        return tab

    def _build_cli_tab(self) -> QWidget:
        tab = QWidget()
        root = QVBoxLayout(tab)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(12)

        root.addWidget(
            self._make_section_header(
                "CLI info",
                "Copy-ready commands generated from the paths you choose here. The guide stays aligned with the real CLI behavior: `inspect` is read-only, `create` writes the lite ZIP run, `priority` writes sidecars only, and `analyse-priority` reads an existing compact priority CSV.",
            )
        )

        scroll = QScrollArea()
        scroll.setObjectName("cliGuideScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        container = QWidget()
        content = QVBoxLayout(container)
        content.setContentsMargins(0, 0, 0, 0)
        content.setSpacing(12)
        content.addWidget(self._build_cli_workspace_card())
        content.addWidget(self._build_cli_summary_card())

        for index, item in enumerate(CLI_GUIDE_SECTIONS):
            content.addWidget(self._build_cli_command_card(item, expanded=index < 2))

        content.addStretch(1)
        scroll.setWidget(container)
        root.addWidget(scroll, 1)
        return tab

    def _build_cli_workspace_card(self) -> QWidget:
        frame = QFrame()
        frame.setObjectName("panel")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(12)

        eyebrow = QLabel("Live command preset")
        eyebrow.setObjectName("eyebrow")
        title = QLabel("Build targeted commands")
        title.setObjectName("sectionTitle")
        body = QLabel(
            "Fill in the paths once and the command cards below update immediately. These fields only shape the command examples; they do not start a run."
        )
        body.setObjectName("muted")
        body.setWordWrap(True)
        layout.addWidget(eyebrow)
        layout.addWidget(title)
        layout.addWidget(body)

        self.cli_profile_name_edit.setPlaceholderText("Friendly preset name")
        self.cli_database_path_edit.setPlaceholderText("/path/to/original_database.zip")
        self.cli_methods_path_edit.setPlaceholderText("/path/to/methods.zip or folder")
        self.cli_output_path_edit.setPlaceholderText("/path/to/output_dir")
        self.cli_priority_csv_path_edit.setPlaceholderText("/path/to/lcia_flow_priority.csv")
        self.cli_metadata_json_path_edit.setPlaceholderText("/path/to/lcia_flow_priority_metadata.json")

        form = QFormLayout()
        form.setSpacing(10)
        form.addRow("Preset name", self.cli_profile_name_edit)
        form.addRow("Database archive", self.cli_database_path_edit)
        form.addRow("Methods archive or folder", self.cli_methods_path_edit)
        form.addRow("Output folder", self.cli_output_path_edit)
        form.addRow("Priority CSV", self.cli_priority_csv_path_edit)
        form.addRow("Metadata JSON", self.cli_metadata_json_path_edit)
        layout.addLayout(form)

        actions = QHBoxLayout()
        sync_reduction = QPushButton("Use reduction paths")
        sync_reduction.setObjectName("secondary")
        sync_reduction.clicked.connect(self._sync_cli_info_fields_from_forms)
        sync_priority = QPushButton("Use priority paths")
        sync_priority.setObjectName("secondary")
        sync_priority.clicked.connect(self._sync_cli_info_priority_fields_from_forms)
        copy_all = QPushButton("Copy all commands")
        copy_all.setObjectName("secondary")
        copy_all.clicked.connect(self.copy_all_cli_commands)
        actions.addWidget(sync_reduction)
        actions.addWidget(sync_priority)
        actions.addWidget(copy_all)
        actions.addStretch(1)
        layout.addLayout(actions)

        self.cli_copy_status_label = QLabel("Commands refresh as you type.")
        self.cli_copy_status_label.setObjectName("muted")
        self.cli_copy_status_label.setWordWrap(True)
        layout.addWidget(self.cli_copy_status_label)

        for line_edit in (
            self.cli_profile_name_edit,
            self.cli_database_path_edit,
            self.cli_methods_path_edit,
            self.cli_output_path_edit,
            self.cli_priority_csv_path_edit,
            self.cli_metadata_json_path_edit,
        ):
            line_edit.textChanged.connect(self._refresh_cli_commands)

        return frame

    def _build_cli_summary_card(self) -> QWidget:
        frame = QFrame()
        frame.setObjectName("panel")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)

        title = QLabel("Recommended order")
        title.setObjectName("sectionTitle")
        title.setStyleSheet("font-size: 16px;")
        body = QLabel(
            "1. Run `inspect` first to confirm whether LCIA methods are already embedded.\n"
            "2. Run `create` when you need the lite JSON-LD ZIP and validation artefacts.\n"
            "3. Run `priority` when you need only LCIA-critical sidecars for transfer or mapping repair.\n"
            "4. Run `analyse-priority` against an existing compact priority CSV.\n"
            "5. Repeat `--select-flow-name` when a flow name contains commas."
        )
        body.setObjectName("muted")
        body.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(body)
        return frame

    def _build_cli_command_card(self, item: dict[str, object], *, expanded: bool) -> QWidget:
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        command_box = QPlainTextEdit()
        command_box.setReadOnly(True)
        command_box.setLineWrapMode(QPlainTextEdit.NoWrap)
        command_box.setMinimumHeight(132)
        command_box.setMaximumHeight(172)

        title = str(item["title"])
        copy_button = QPushButton("Copy command")
        copy_button.setObjectName("secondary")
        copy_button.clicked.connect(lambda: self.copy_cli_command(title))

        button_row = QHBoxLayout()
        button_row.setContentsMargins(0, 0, 0, 0)
        button_row.addWidget(copy_button)
        button_row.addStretch(1)

        notes = QLabel()
        notes.setObjectName("muted")
        notes.setWordWrap(True)

        layout.addWidget(command_box)
        layout.addLayout(button_row)
        layout.addWidget(notes)

        self.cli_command_boxes[title] = command_box
        self.cli_note_labels[title] = notes
        return AnimatedCard(title, str(item["summary"]), content, expanded=expanded)

    def _sync_cli_info_fields_from_forms(self) -> None:
        self.cli_database_path_edit.setText(self.database_edit.text().strip())
        self.cli_methods_path_edit.setText(self.methods_edit.text().strip())
        self.cli_output_path_edit.setText(self.output_edit.text().strip())
        if self.cli_copy_status_label is not None:
            self.cli_copy_status_label.setText("Loaded the current Reduction tab paths into the CLI guide.")

    def _sync_cli_info_priority_fields_from_forms(self) -> None:
        if self.priority_database_edit.text().strip():
            self.cli_database_path_edit.setText(self.priority_database_edit.text().strip())
        if self.priority_methods_edit.text().strip():
            self.cli_methods_path_edit.setText(self.priority_methods_edit.text().strip())
        if self.priority_output_edit.text().strip():
            self.cli_output_path_edit.setText(self.priority_output_edit.text().strip())
        if self.cli_copy_status_label is not None:
            self.cli_copy_status_label.setText("Loaded the current Flow priority tab paths into the CLI guide.")

    def _refresh_cli_commands(self) -> None:
        profile_name = self.cli_profile_name_edit.text().strip() or "my_database"
        database = self._shell_arg(self.cli_database_path_edit.text().strip(), "/path/to/original_database.zip")
        methods_value = self.cli_methods_path_edit.text().strip()
        methods_arg = (
            f" \\\n  --methods {self._shell_arg(methods_value, '/path/to/methods.zip')}"
            if methods_value
            else ""
        )
        output_value = self.cli_output_path_edit.text().strip()
        output_dir = self._shell_arg(output_value, "/path/to/output_dir")
        derived_priority_csv = (
            f"{output_value.rstrip('/')}/lcia_flow_priority.csv"
            if output_value
            else "/path/to/lcia_flow_priority.csv"
        )
        derived_metadata_json = (
            f"{output_value.rstrip('/')}/lcia_flow_priority_metadata.json"
            if output_value
            else "/path/to/lcia_flow_priority_metadata.json"
        )
        priority_csv = self._shell_arg(
            self.cli_priority_csv_path_edit.text().strip(),
            derived_priority_csv,
        )
        metadata_json = self._shell_arg(
            self.cli_metadata_json_path_edit.text().strip(),
            derived_metadata_json,
        )

        commands = {
            "1. Inspect Before You Run": (
                f"# Inspect {profile_name} before any write operation\n"
                "lci_reduce inspect \\\n"
                f"  --database {database}{methods_arg}"
            ),
            "2. Create A Lite Database": (
                f"# Create a reduced JSON-LD run for {profile_name}\n"
                "lci_reduce create \\\n"
                f"  --database {database}{methods_arg} \\\n"
                f"  --output {output_dir} \\\n"
                "  --tau 0.95 \\\n"
                "  --method-selection all \\\n"
                "  --uncharacterised-policy keep \\\n"
                "  --strict-units true"
            ),
            "3. Generate LCIA Flow Priority Sidecars": (
                f"# Generate LCIA-critical sidecars for {profile_name}\n"
                "lci_reduce priority \\\n"
                f"  --database {database}{methods_arg} \\\n"
                f"  --output {output_dir} \\\n"
                "  --method-selection all \\\n"
                "  --audit-tau 0.95 0.99 \\\n"
                "  --strict-units true"
            ),
            "4. Analyse An Existing Priority File": (
                f"# Screen a compact priority CSV for {profile_name}\n"
                "lci_reduce analyse-priority \\\n"
                f"  --priority-csv {priority_csv} \\\n"
                f"  --metadata-json {metadata_json} \\\n"
                "  --audit-tau 0.95 \\\n"
                "  --top-n 20 \\\n"
                "  --select-flow-id flow-1 \\\n"
                "  --select-flow-name 'Sulfur dioxide' \\\n"
                "  --output-ranked-csv /path/to/ranked.csv \\\n"
                "  --output-summary-json /path/to/priority_analysis_summary.json"
            ),
            "5. Start The Desktop GUI": "lci_reduce-gui",
            "6. Alternate Entrypoints": (
                f"python -m lci_reduce cli inspect --database {database}\n"
                "python -m lci_reduce gui\n"
                f"python main.py cli analyse-priority --priority-csv {priority_csv}\n"
                "python main.py"
            ),
        }

        for item in CLI_GUIDE_SECTIONS:
            title = str(item["title"])
            if title in self.cli_command_boxes:
                self.cli_command_boxes[title].setPlainText(commands.get(title, str(item["command"])))
            if title in self.cli_note_labels:
                self.cli_note_labels[title].setText("\n".join(f"- {note}" for note in item["notes"]))

    @staticmethod
    def _shell_arg(value: str, fallback: str) -> str:
        return shlex.quote(value or fallback)

    def copy_cli_command(self, title: str) -> None:
        command_box = self.cli_command_boxes.get(title)
        if command_box is None:
            return
        QApplication.clipboard().setText(command_box.toPlainText())
        if self.cli_copy_status_label is not None:
            self.cli_copy_status_label.setText(f"Copied the command for {title}.")

    def copy_all_cli_commands(self) -> None:
        blocks: list[str] = []
        for item in CLI_GUIDE_SECTIONS:
            title = str(item["title"])
            command_box = self.cli_command_boxes.get(title)
            if command_box is None:
                continue
            blocks.append(f"{title}\n{command_box.toPlainText()}")
        QApplication.clipboard().setText("\n\n".join(blocks))
        if self.cli_copy_status_label is not None:
            self.cli_copy_status_label.setText("Copied all CLI guide commands.")

    def _make_chart_view(self) -> QChartView:
        chart = QChart()
        chart.legend().setVisible(True)
        chart.legend().setAlignment(Qt.AlignBottom)
        view = QChartView(chart)
        view.setRenderHint(QPainter.Antialiasing)
        view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        return view

    def append_status(self, text: str) -> None:
        self.status_box.appendPlainText(text)

    def append_priority_status(self, text: str) -> None:
        self.priority_status_box.appendPlainText(text)

    def append_ambiguity_status(self, text: str) -> None:
        self.ambiguity_status_box.appendPlainText(text)

    def append_ecospold_ambiguity_status(self, text: str) -> None:
        self.ecospold_ambiguity_status_box.appendPlainText(text)

    def append_ecospold_status(self, text: str) -> None:
        self.ecospold_status_box.appendPlainText(text)

    def append_ecospold_priority_status(self, text: str) -> None:
        self.ecospold_priority_status_box.appendPlainText(text)

    def append_diagnostic_status(self, text: str) -> None:
        self.diagnostic_status_box.appendPlainText(text)

    def set_output_paths(self, lines: list[str]) -> None:
        self.output_box.setPlainText("\n".join(line for line in lines if line))

    def set_priority_output_paths(self, lines: list[str]) -> None:
        self.priority_output_box.setPlainText("\n".join(line for line in lines if line))

    def set_ambiguity_output_paths(self, lines: list[str]) -> None:
        self.ambiguity_output_box.setPlainText("\n".join(line for line in lines if line))

    def set_ecospold_ambiguity_output_paths(self, lines: list[str]) -> None:
        self.ecospold_ambiguity_output_box.setPlainText("\n".join(line for line in lines if line))

    def set_ecospold_output_paths(self, lines: list[str]) -> None:
        self.ecospold_output_box.setPlainText("\n".join(line for line in lines if line))

    def set_ecospold_priority_output_paths(self, lines: list[str]) -> None:
        self.ecospold_priority_output_box.setPlainText("\n".join(line for line in lines if line))

    def set_diagnostic_output_paths(self, lines: list[str]) -> None:
        self.diagnostic_output_box.setPlainText("\n".join(line for line in lines if line))

    def reset_run_metrics(self) -> None:
        self.stage_value.setText("Idle")
        self.process_value.setText("0 / 0")
        self.exchange_value.setText("0 / 0")
        self.current_process_value.setText("Ready")
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)

    def reset_priority_metrics(self) -> None:
        self.priority_stage_value.setText("Idle")
        self.priority_process_value.setText("0 / 0")
        self.priority_current_process_value.setText("Ready")
        self.priority_progress_bar.setRange(0, 1)
        self.priority_progress_bar.setValue(0)

    def reset_ambiguity_metrics(self) -> None:
        self.ambiguity_stage_value.setText("Idle")
        self.ambiguity_process_value.setText("0 / 0")
        self.ambiguity_current_process_value.setText("Ready")
        self.ambiguity_progress_bar.setRange(0, 1)
        self.ambiguity_progress_bar.setValue(0)

    def reset_ecospold_ambiguity_metrics(self) -> None:
        self.ecospold_ambiguity_stage_value.setText("Idle")
        self.ecospold_ambiguity_process_value.setText("0 / 0")
        self.ecospold_ambiguity_current_process_value.setText("Ready")
        self.ecospold_ambiguity_progress_bar.setRange(0, 1)
        self.ecospold_ambiguity_progress_bar.setValue(0)

    def reset_ecospold_metrics(self) -> None:
        self.ecospold_stage_value.setText("Idle")
        self.ecospold_process_value.setText("0 / 0")
        self.ecospold_exchange_value.setText("0 / 0")
        self.ecospold_current_process_value.setText("Ready")
        self.ecospold_progress_bar.setRange(0, 1)
        self.ecospold_progress_bar.setValue(0)

    def reset_ecospold_priority_metrics(self) -> None:
        self.ecospold_priority_stage_value.setText("Idle")
        self.ecospold_priority_process_value.setText("0 / 0")
        self.ecospold_priority_current_process_value.setText("Ready")
        self.ecospold_priority_progress_bar.setRange(0, 1)
        self.ecospold_priority_progress_bar.setValue(0)

    def reset_diagnostic_metrics(self) -> None:
        self.diagnostic_stage_value.setText("Idle")
        self.diagnostic_runtime_value.setText("-")
        self.diagnostic_process_value.setText("Select a process")
        self.diagnostic_protected_value.setText("-")
        self.diagnostic_greedy_coverage_value.setText("-")
        self.diagnostic_exact_coverage_value.setText("-")
        self.diagnostic_certificate_value.setText("-")
        self.diagnostic_progress_bar.setRange(0, 1)
        self.diagnostic_progress_bar.setValue(0)

    def _set_run_controls_enabled(self, enabled: bool) -> None:
        if self.inspect_button is not None:
            self.inspect_button.setDisabled(not enabled)
        if self.create_button is not None:
            self.create_button.setDisabled(not enabled)
        if self.priority_button is not None:
            self.priority_button.setDisabled(not enabled)
        if self.ambiguity_button is not None:
            self.ambiguity_button.setDisabled(not enabled)
        if self.ecospold_ambiguity_button is not None:
            self.ecospold_ambiguity_button.setDisabled(not enabled)
        if self.ecospold_create_button is not None:
            self.ecospold_create_button.setDisabled(not enabled)
        if self.ecospold_priority_button is not None:
            self.ecospold_priority_button.setDisabled(not enabled)
        if self.diagnostic_run_button is not None:
            self.diagnostic_run_button.setDisabled(not enabled)
        if self.use_database_methods_button is not None:
            self.use_database_methods_button.setDisabled((not enabled) or not self.database_has_impact_methods)
        if self.priority_use_database_methods_button is not None:
            self.priority_use_database_methods_button.setDisabled(
                (not enabled) or not self.priority_database_has_impact_methods
            )
        if self.ambiguity_use_database_methods_button is not None:
            self.ambiguity_use_database_methods_button.setDisabled(
                (not enabled) or not self.ambiguity_database_has_impact_methods
            )
        if self.ecospold_ambiguity_use_database_methods_button is not None:
            self.ecospold_ambiguity_use_database_methods_button.setDisabled(
                (not enabled) or not self.ecospold_ambiguity_database_has_impact_methods
            )
        if self.diagnostic_use_database_methods_button is not None:
            self.diagnostic_use_database_methods_button.setDisabled(
                (not enabled) or not self.diagnostic_database_has_impact_methods
            )

    def set_busy(self, busy: bool, message: str) -> None:
        self._set_run_controls_enabled(not busy)
        if busy:
            self.stage_value.setText("Running")
            self.progress_bar.setRange(0, 0)
        else:
            self.progress_bar.setRange(0, 1)
            self.progress_bar.setValue(1)
        self.append_status(message)

    def set_priority_busy(self, busy: bool, message: str) -> None:
        self._set_run_controls_enabled(not busy)
        if busy:
            self.priority_stage_value.setText("Running")
            self.priority_progress_bar.setRange(0, 0)
        else:
            self.priority_progress_bar.setRange(0, 1)
            self.priority_progress_bar.setValue(1)
        self.append_priority_status(message)

    def set_ambiguity_busy(self, busy: bool, message: str) -> None:
        self._set_run_controls_enabled(not busy)
        if busy:
            self.ambiguity_stage_value.setText("Running")
            self.ambiguity_progress_bar.setRange(0, 0)
        else:
            self.ambiguity_progress_bar.setRange(0, 1)
            self.ambiguity_progress_bar.setValue(1)
        self.append_ambiguity_status(message)

    def set_ecospold_ambiguity_busy(self, busy: bool, message: str) -> None:
        self._set_run_controls_enabled(not busy)
        if busy:
            self.ecospold_ambiguity_stage_value.setText("Running")
            self.ecospold_ambiguity_progress_bar.setRange(0, 0)
        else:
            self.ecospold_ambiguity_progress_bar.setRange(0, 1)
            self.ecospold_ambiguity_progress_bar.setValue(1)
        self.append_ecospold_ambiguity_status(message)

    def set_ecospold_busy(self, busy: bool, message: str) -> None:
        self._set_run_controls_enabled(not busy)
        if busy:
            self.ecospold_stage_value.setText("Running")
            self.ecospold_progress_bar.setRange(0, 0)
        else:
            self.ecospold_progress_bar.setRange(0, 1)
            self.ecospold_progress_bar.setValue(1)
        self.append_ecospold_status(message)

    def set_ecospold_priority_busy(self, busy: bool, message: str) -> None:
        self._set_run_controls_enabled(not busy)
        if busy:
            self.ecospold_priority_stage_value.setText("Running")
            self.ecospold_priority_progress_bar.setRange(0, 0)
        else:
            self.ecospold_priority_progress_bar.setRange(0, 1)
            self.ecospold_priority_progress_bar.setValue(1)
        self.append_ecospold_priority_status(message)

    def set_diagnostic_busy(self, busy: bool, message: str) -> None:
        self._set_run_controls_enabled(not busy)
        if busy:
            self.diagnostic_stage_value.setText("Running")
            self.diagnostic_progress_bar.setRange(0, 0)
        else:
            self.diagnostic_progress_bar.setRange(0, 1)
            self.diagnostic_progress_bar.setValue(1)
        self.append_diagnostic_status(message)

    def _set_database_methods_hint(self, result: dict | None) -> None:
        if not result:
            self.database_has_impact_methods = False
            self.database_methods_label.setText("Select a database archive to check for embedded LCIA methods.")
            if self.use_database_methods_button is not None:
                self.use_database_methods_button.setEnabled(False)
            return
        if result.get("database_contains_impact_methods"):
            self.database_has_impact_methods = True
            self.database_methods_label.setText(
                "The selected database already contains "
                f"{result.get('database_lcia_methods', 0)} impact methods and "
                f"{result.get('database_lcia_categories', 0)} impact categories."
            )
            if self.use_database_methods_button is not None and self._reduction_thread is None:
                self.use_database_methods_button.setEnabled(True)
        else:
            self.database_has_impact_methods = False
            self.database_methods_label.setText(
                "No embedded impact methods were found. Provide an optional methods archive or folder if required."
            )
            if self.use_database_methods_button is not None:
                self.use_database_methods_button.setEnabled(False)

    def _set_priority_database_methods_hint(self, result: dict | None) -> None:
        if not result:
            self.priority_database_has_impact_methods = False
            self.priority_database_methods_label.setText(
                "Select a database archive to check for embedded LCIA methods."
            )
            if self.priority_use_database_methods_button is not None:
                self.priority_use_database_methods_button.setEnabled(False)
            return
        if result.get("database_contains_impact_methods"):
            self.priority_database_has_impact_methods = True
            self.priority_database_methods_label.setText(
                "The selected database already contains "
                f"{result.get('database_lcia_methods', 0)} impact methods and "
                f"{result.get('database_lcia_categories', 0)} impact categories."
            )
            if self.priority_use_database_methods_button is not None and self._reduction_thread is None:
                self.priority_use_database_methods_button.setEnabled(True)
        else:
            self.priority_database_has_impact_methods = False
            self.priority_database_methods_label.setText(
                "No embedded impact methods were found. Provide an optional methods archive or folder if required."
            )
            if self.priority_use_database_methods_button is not None:
                self.priority_use_database_methods_button.setEnabled(False)
            return

    def _set_ambiguity_database_methods_hint(self, result: dict | None) -> None:
        if not result:
            self.ambiguity_database_has_impact_methods = False
            self.ambiguity_database_methods_label.setText(
                "Select a database archive to scan ambiguity records without reducing the database."
            )
            if self.ambiguity_use_database_methods_button is not None:
                self.ambiguity_use_database_methods_button.setEnabled(False)
            return
        if result.get("database_contains_impact_methods"):
            self.ambiguity_database_has_impact_methods = True
            self.ambiguity_database_methods_label.setText(
                "The selected database already contains "
                f"{result.get('database_lcia_methods', 0)} impact methods and "
                f"{result.get('database_lcia_categories', 0)} impact categories."
            )
            if self.ambiguity_use_database_methods_button is not None and self._reduction_thread is None:
                self.ambiguity_use_database_methods_button.setEnabled(True)
        else:
            self.ambiguity_database_has_impact_methods = False
            self.ambiguity_database_methods_label.setText(
                "No embedded impact methods were found. Provide an optional methods archive or folder if required."
            )
            if self.ambiguity_use_database_methods_button is not None:
                self.ambiguity_use_database_methods_button.setEnabled(False)
            return

    def _set_ecospold_ambiguity_database_methods_hint(self, result: dict | None) -> None:
        if not result:
            self.ecospold_ambiguity_database_has_impact_methods = False
            self.ecospold_ambiguity_database_methods_label.setText(
                "Select an EcoSpold1 process archive or folder to scan ambiguity records without reducing the database."
            )
            if self.ecospold_ambiguity_use_database_methods_button is not None:
                self.ecospold_ambiguity_use_database_methods_button.setEnabled(False)
            return
        if result.get("database_contains_impact_methods"):
            self.ecospold_ambiguity_database_has_impact_methods = True
            self.ecospold_ambiguity_database_methods_label.setText(
                "The selected EcoSpold1 archive already contains "
                f"{result.get('database_lcia_methods', 0)} impact methods and "
                f"{result.get('database_lcia_categories', 0)} impact categories."
            )
            if self.ecospold_ambiguity_use_database_methods_button is not None and self._reduction_thread is None:
                self.ecospold_ambiguity_use_database_methods_button.setEnabled(True)
        else:
            self.ecospold_ambiguity_database_has_impact_methods = False
            self.ecospold_ambiguity_database_methods_label.setText(
                "No embedded impact methods were found. Provide an optional methods archive or folder if required."
            )
            if self.ecospold_ambiguity_use_database_methods_button is not None:
                self.ecospold_ambiguity_use_database_methods_button.setEnabled(False)
            return

    def _set_diagnostic_database_methods_hint(self, result: dict | None) -> None:
        if not result:
            self.diagnostic_database_has_impact_methods = False
            self.diagnostic_database_methods_label.setText(
                "Select a database archive to check for embedded LCIA methods."
            )
            if self.diagnostic_use_database_methods_button is not None:
                self.diagnostic_use_database_methods_button.setEnabled(False)
            return
        if result.get("database_contains_impact_methods"):
            self.diagnostic_database_has_impact_methods = True
            self.diagnostic_database_methods_label.setText(
                "The selected database already contains "
                f"{result.get('database_lcia_methods', 0)} impact methods and "
                f"{result.get('database_lcia_categories', 0)} impact categories."
            )
            if self.diagnostic_use_database_methods_button is not None and self._reduction_thread is None:
                self.diagnostic_use_database_methods_button.setEnabled(True)
        else:
            self.diagnostic_database_has_impact_methods = False
            self.diagnostic_database_methods_label.setText(
                "No embedded impact methods were found. Provide an optional methods archive or folder if required."
            )
            if self.diagnostic_use_database_methods_button is not None:
                self.diagnostic_use_database_methods_button.setEnabled(False)
            return
        if result.get("database_contains_impact_methods"):
            self.priority_database_has_impact_methods = True
            self.priority_database_methods_label.setText(
                "The selected database already contains "
                f"{result.get('database_lcia_methods', 0)} impact methods and "
                f"{result.get('database_lcia_categories', 0)} impact categories."
            )
            if self.priority_use_database_methods_button is not None and self._reduction_thread is None:
                self.priority_use_database_methods_button.setEnabled(True)
        else:
            self.priority_database_has_impact_methods = False
            self.priority_database_methods_label.setText(
                "No embedded impact methods were found. Provide an optional methods archive or folder if required."
            )
            if self.priority_use_database_methods_button is not None:
                self.priority_use_database_methods_button.setEnabled(False)

    def pick_database(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select database archive",
            "",
            "Archive files (*.zip *.zolca)",
        )
        if path:
            self.database_edit.setText(path)
            if self._reduction_thread is None:
                self.inspect_database_methods()

    def pick_methods(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select methods folder")
        if not path:
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Select methods archive",
                "",
                "Archive files (*.zip *.zolca)",
            )
        if path:
            self.methods_edit.setText(path)

    def pick_output(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select output folder")
        if path:
            self.output_edit.setText(path)

    def _pick_folder_or_archive(self, folder_title: str, archive_title: str) -> str:
        path = QFileDialog.getExistingDirectory(self, folder_title)
        if path:
            return path
        path, _ = QFileDialog.getOpenFileName(
            self,
            archive_title,
            "",
            "Archive files (*.zip *.xml *.spold)",
        )
        return path

    def pick_priority_database(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select database archive",
            "",
            "Archive files (*.zip *.zolca)",
        )
        if path:
            self.priority_database_edit.setText(path)
            if self._reduction_thread is None:
                self.inspect_priority_database_methods()

    def pick_priority_methods(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select methods folder")
        if not path:
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Select methods archive",
                "",
                "Archive files (*.zip *.zolca)",
            )
        if path:
            self.priority_methods_edit.setText(path)

    def pick_priority_output(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select output folder")
        if path:
            self.priority_output_edit.setText(path)

    def pick_ambiguity_database(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select database archive",
            "",
            "Archive files (*.zip *.zolca)",
        )
        if path:
            self.ambiguity_database_edit.setText(path)
            if self._reduction_thread is None:
                self.inspect_ambiguity_database_methods()

    def pick_ambiguity_methods(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select methods folder")
        if not path:
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Select methods archive",
                "",
                "Archive files (*.zip *.zolca)",
            )
        if path:
            self.ambiguity_methods_edit.setText(path)

    def pick_ambiguity_output(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select output folder")
        if path:
            self.ambiguity_output_edit.setText(path)

    def pick_ecospold_ambiguity_database(self) -> None:
        path = self._pick_folder_or_archive("Select EcoSpold1 process folder", "Select EcoSpold1 process archive")
        if path:
            self.ecospold_ambiguity_database_edit.setText(path)
            if self._reduction_thread is None:
                self.inspect_ecospold_ambiguity_database_methods()

    def pick_ecospold_ambiguity_methods(self) -> None:
        path = self._pick_folder_or_archive("Select EcoSpold1 impact-method folder", "Select EcoSpold1 impact-method archive")
        if path:
            self.ecospold_ambiguity_methods_edit.setText(path)

    def pick_ecospold_ambiguity_output(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select output folder")
        if path:
            self.ecospold_ambiguity_output_edit.setText(path)

    def pick_ecospold_database(self) -> None:
        path = self._pick_folder_or_archive("Select EcoSpold1 process folder", "Select EcoSpold1 process archive")
        if path:
            self.ecospold_database_edit.setText(path)

    def pick_ecospold_methods(self) -> None:
        path = self._pick_folder_or_archive("Select EcoSpold1 impact-method folder", "Select EcoSpold1 impact-method archive")
        if path:
            self.ecospold_methods_edit.setText(path)

    def pick_ecospold_output(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select output folder")
        if path:
            self.ecospold_output_edit.setText(path)

    def pick_ecospold_priority_database(self) -> None:
        path = self._pick_folder_or_archive("Select EcoSpold1 process folder", "Select EcoSpold1 process archive")
        if path:
            self.ecospold_priority_database_edit.setText(path)

    def pick_ecospold_priority_methods(self) -> None:
        path = self._pick_folder_or_archive("Select EcoSpold1 impact-method folder", "Select EcoSpold1 impact-method archive")
        if path:
            self.ecospold_priority_methods_edit.setText(path)

    def pick_ecospold_priority_output(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select output folder")
        if path:
            self.ecospold_priority_output_edit.setText(path)

    def pick_diagnostic_database(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select database archive",
            "",
            "Archive files (*.zip *.zolca)",
        )
        if path:
            self.diagnostic_database_edit.setText(path)
            if self._reduction_thread is None:
                self.inspect_diagnostic_database_methods()

    def pick_diagnostic_methods(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select methods folder")
        if not path:
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Select methods archive",
                "",
                "Archive files (*.zip *.zolca)",
            )
        if path:
            self.diagnostic_methods_edit.setText(path)

    def pick_diagnostic_output(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select output folder")
        if path:
            self.diagnostic_output_edit.setText(path)

    def use_database_methods(self) -> None:
        self.methods_edit.clear()
        self.append_status("Using LCIA methods contained in the database. External methods input cleared.")

    def use_priority_database_methods(self) -> None:
        self.priority_methods_edit.clear()
        self.append_priority_status("Using LCIA methods contained in the database. External methods input cleared.")

    def use_ambiguity_database_methods(self) -> None:
        self.ambiguity_methods_edit.clear()
        self.append_ambiguity_status("Using LCIA methods contained in the database. External methods input cleared.")

    def use_ecospold_ambiguity_database_methods(self) -> None:
        self.ecospold_ambiguity_methods_edit.clear()
        self.append_ecospold_ambiguity_status(
            "Using LCIA methods contained in the database. External methods input cleared."
        )

    def use_diagnostic_database_methods(self) -> None:
        self.diagnostic_methods_edit.clear()
        self.append_diagnostic_status(
            "Using LCIA methods contained in the database. External methods input cleared."
        )

    def _start_reduction_worker(self, worker: QObject, *, mode: str) -> None:
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.done.connect(thread.quit)
        worker.done.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_reduction_worker_finished)
        self._reduction_thread = thread
        self._reduction_worker = worker
        self._reduction_mode = mode
        thread.start()

    def _clear_reduction_worker(self) -> None:
        self._reduction_thread = None
        self._reduction_worker = None
        self._reduction_mode = ""

    @Slot()
    def _on_reduction_worker_finished(self) -> None:
        mode = self._reduction_mode
        self._clear_reduction_worker()
        message = "Run finished."
        if mode == "hint":
            message = "Database method scan finished."
        elif mode == "inspect":
            message = "Inspect finished."
        elif mode == "create":
            message = "Create finished."
            self.set_busy(False, message)
            return
        elif mode == "priority_hint":
            message = "Database method scan finished."
            self.set_priority_busy(False, message)
            return
        elif mode == "priority":
            message = "Flow-priority audit finished."
            self.set_priority_busy(False, message)
            return
        elif mode == "ambiguity_hint":
            message = "Database method scan finished."
            self.set_ambiguity_busy(False, message)
            return
        elif mode == "ambiguity":
            message = "Ambiguity exploration finished."
            self.set_ambiguity_busy(False, message)
            return
        elif mode == "ecospold_ambiguity_hint":
            message = "EcoSpold1 database method scan finished."
            self.set_ecospold_ambiguity_busy(False, message)
            return
        elif mode == "ecospold_ambiguity":
            message = "EcoSpold1 ambiguity exploration finished."
            self.set_ecospold_ambiguity_busy(False, message)
            return
        elif mode == "ecospold_create":
            message = "EcoSpold reduction finished."
            self.set_ecospold_busy(False, message)
            return
        elif mode == "ecospold_priority":
            message = "EcoSpold flow-priority audit finished."
            self.set_ecospold_priority_busy(False, message)
            return
        elif mode == "diagnostic_hint":
            message = "Database method scan finished."
            self.set_diagnostic_busy(False, message)
            return
        elif mode == "diagnostic":
            message = "Greedy vs exact diagnostic finished."
            self.set_diagnostic_busy(False, message)
            return
        self.set_busy(False, message)

    def inspect_database_methods(self) -> None:
        database = self.database_edit.text().strip()
        if not database or self._reduction_thread is not None:
            return
        self.set_busy(True, "Inspecting database methods...")
        worker = InspectWorker(database, None)
        worker.finished.connect(self._handle_database_method_hint)
        worker.failed.connect(self._handle_worker_failure)
        self._start_reduction_worker(worker, mode="hint")

    def inspect_priority_database_methods(self) -> None:
        database = self.priority_database_edit.text().strip()
        if not database or self._reduction_thread is not None:
            return
        self.set_priority_busy(True, "Inspecting database methods...")
        worker = InspectWorker(database, None)
        worker.finished.connect(self._handle_priority_database_method_hint)
        worker.failed.connect(self._handle_worker_failure)
        self._start_reduction_worker(worker, mode="priority_hint")

    def inspect_ambiguity_database_methods(self) -> None:
        database = self.ambiguity_database_edit.text().strip()
        if not database or self._reduction_thread is not None:
            return
        self.set_ambiguity_busy(True, "Inspecting database methods...")
        worker = InspectWorker(database, None)
        worker.finished.connect(self._handle_ambiguity_database_method_hint)
        worker.failed.connect(self._handle_worker_failure)
        self._start_reduction_worker(worker, mode="ambiguity_hint")

    def inspect_ecospold_ambiguity_database_methods(self) -> None:
        database = self.ecospold_ambiguity_database_edit.text().strip()
        if not database or self._reduction_thread is not None:
            return
        self.set_ecospold_ambiguity_busy(True, "Inspecting EcoSpold1 methods...")
        worker = InspectWorker(database, None)
        worker.finished.connect(self._handle_ecospold_ambiguity_database_method_hint)
        worker.failed.connect(self._handle_worker_failure)
        self._start_reduction_worker(worker, mode="ecospold_ambiguity_hint")

    def inspect_diagnostic_database_methods(self) -> None:
        database = self.diagnostic_database_edit.text().strip()
        if not database or self._reduction_thread is not None:
            return
        self.set_diagnostic_busy(True, "Inspecting database methods...")
        worker = InspectWorker(database, None)
        worker.finished.connect(self._handle_diagnostic_database_method_hint)
        worker.failed.connect(self._handle_worker_failure)
        self._start_reduction_worker(worker, mode="diagnostic_hint")

    @Slot(object)
    def _handle_database_method_hint(self, result: object) -> None:
        data = result if isinstance(result, dict) else None
        self._set_database_methods_hint(data)
        if data and data.get("database_methods_hint"):
            self.append_status(data["database_methods_hint"])

    @Slot(object)
    def _handle_priority_database_method_hint(self, result: object) -> None:
        data = result if isinstance(result, dict) else None
        self._set_priority_database_methods_hint(data)
        if data and data.get("database_methods_hint"):
            self.append_priority_status(data["database_methods_hint"])

    @Slot(object)
    def _handle_ambiguity_database_method_hint(self, result: object) -> None:
        data = result if isinstance(result, dict) else None
        self._set_ambiguity_database_methods_hint(data)
        if data and data.get("database_methods_hint"):
            self.append_ambiguity_status(data["database_methods_hint"])

    @Slot(object)
    def _handle_ecospold_ambiguity_database_method_hint(self, result: object) -> None:
        data = result if isinstance(result, dict) else None
        self._set_ecospold_ambiguity_database_methods_hint(data)
        if data and data.get("database_methods_hint"):
            self.append_ecospold_ambiguity_status(data["database_methods_hint"])

    @Slot(object)
    def _handle_diagnostic_database_method_hint(self, result: object) -> None:
        data = result if isinstance(result, dict) else None
        self._set_diagnostic_database_methods_hint(data)
        if data and data.get("database_methods_hint"):
            self.append_diagnostic_status(data["database_methods_hint"])

    @Slot(str)
    def _handle_worker_failure(self, message: str) -> None:
        QMessageBox.critical(self, "Run failed", message)
        if self._reduction_mode in {"priority", "priority_hint"}:
            self.append_priority_status(f"Run failed: {message}")
        elif self._reduction_mode in {"ambiguity", "ambiguity_hint"}:
            self.append_ambiguity_status(f"Run failed: {message}")
        elif self._reduction_mode in {"ecospold_ambiguity", "ecospold_ambiguity_hint"}:
            self.append_ecospold_ambiguity_status(f"Run failed: {message}")
        elif self._reduction_mode == "ecospold_create":
            self.append_ecospold_status(f"Run failed: {message}")
        elif self._reduction_mode == "ecospold_priority":
            self.append_ecospold_priority_status(f"Run failed: {message}")
        elif self._reduction_mode in {"diagnostic", "diagnostic_hint"}:
            self.append_diagnostic_status(f"Run failed: {message}")
        else:
            self.append_status(f"Run failed: {message}")

    def update_create_progress(self, update: CreateProgressUpdate) -> None:
        stage_total = max(update.stage_total or update.total or 1, 1)
        stage_current = min(max(update.stage_current or update.current, 0), stage_total)
        self.stage_value.setText(f"{stage_current} / {stage_total}")

        if update.process_total:
            current = min(max(update.process_current or 0, 0), update.process_total)
            self.process_value.setText(f"{current} / {update.process_total}")
            self.progress_bar.setRange(0, update.process_total)
            self.progress_bar.setValue(current)
        else:
            self.process_value.setText("0 / 0")
            self.progress_bar.setRange(0, stage_total)
            self.progress_bar.setValue(stage_current)

        removed = update.n_elementary_removed or 0
        before = update.n_elementary_before or 0
        self.exchange_value.setText(f"{removed} / {before}")
        self.current_process_value.setText(update.process_name or update.message)
        self.append_status(update.message)

    def update_priority_progress(self, update: CreateProgressUpdate) -> None:
        stage_total = max(update.stage_total or update.total or 1, 1)
        stage_current = min(max(update.stage_current or update.current, 0), stage_total)
        self.priority_stage_value.setText(f"{stage_current} / {stage_total}")

        if update.process_total:
            current = min(max(update.process_current or 0, 0), update.process_total)
            self.priority_process_value.setText(f"{current} / {update.process_total}")
            self.priority_progress_bar.setRange(0, update.process_total)
            self.priority_progress_bar.setValue(current)
        else:
            self.priority_process_value.setText("0 / 0")
            self.priority_progress_bar.setRange(0, stage_total)
            self.priority_progress_bar.setValue(stage_current)

        self.priority_current_process_value.setText(update.process_name or update.message)
        self.append_priority_status(update.message)

    def update_ambiguity_progress(self, update: CreateProgressUpdate) -> None:
        stage_total = max(update.stage_total or update.total or 1, 1)
        stage_current = min(max(update.stage_current or update.current, 0), stage_total)
        self.ambiguity_stage_value.setText(f"{stage_current} / {stage_total}")

        if update.process_total:
            current = min(max(update.process_current or 0, 0), update.process_total)
            self.ambiguity_process_value.setText(f"{current} / {update.process_total}")
            self.ambiguity_progress_bar.setRange(0, update.process_total)
            self.ambiguity_progress_bar.setValue(current)
        else:
            self.ambiguity_process_value.setText("0 / 0")
            self.ambiguity_progress_bar.setRange(0, stage_total)
            self.ambiguity_progress_bar.setValue(stage_current)

        self.ambiguity_current_process_value.setText(update.process_name or update.message)
        self.append_ambiguity_status(update.message)

    def update_ecospold_ambiguity_progress(self, update: CreateProgressUpdate) -> None:
        stage_total = max(update.stage_total or update.total or 1, 1)
        stage_current = min(max(update.stage_current or update.current, 0), stage_total)
        self.ecospold_ambiguity_stage_value.setText(f"{stage_current} / {stage_total}")

        if update.process_total:
            current = min(max(update.process_current or 0, 0), update.process_total)
            self.ecospold_ambiguity_process_value.setText(f"{current} / {update.process_total}")
            self.ecospold_ambiguity_progress_bar.setRange(0, update.process_total)
            self.ecospold_ambiguity_progress_bar.setValue(current)
        else:
            self.ecospold_ambiguity_process_value.setText("0 / 0")
            self.ecospold_ambiguity_progress_bar.setRange(0, stage_total)
            self.ecospold_ambiguity_progress_bar.setValue(stage_current)

        self.ecospold_ambiguity_current_process_value.setText(update.process_name or update.message)
        self.append_ecospold_ambiguity_status(update.message)

    def update_ecospold_progress(self, update: CreateProgressUpdate) -> None:
        stage_total = max(update.stage_total or update.total or 1, 1)
        stage_current = min(max(update.stage_current or update.current, 0), stage_total)
        self.ecospold_stage_value.setText(f"{stage_current} / {stage_total}")

        if update.process_total:
            current = min(max(update.process_current or 0, 0), update.process_total)
            self.ecospold_process_value.setText(f"{current} / {update.process_total}")
            self.ecospold_progress_bar.setRange(0, update.process_total)
            self.ecospold_progress_bar.setValue(current)
        else:
            self.ecospold_process_value.setText("0 / 0")
            self.ecospold_progress_bar.setRange(0, stage_total)
            self.ecospold_progress_bar.setValue(stage_current)

        removed = update.n_elementary_removed or 0
        before = update.n_elementary_before or 0
        self.ecospold_exchange_value.setText(f"{removed} / {before}")
        self.ecospold_current_process_value.setText(update.process_name or update.message)
        self.append_ecospold_status(update.message)

    def update_ecospold_priority_progress(self, update: CreateProgressUpdate) -> None:
        stage_total = max(update.stage_total or update.total or 1, 1)
        stage_current = min(max(update.stage_current or update.current, 0), stage_total)
        self.ecospold_priority_stage_value.setText(f"{stage_current} / {stage_total}")

        if update.process_total:
            current = min(max(update.process_current or 0, 0), update.process_total)
            self.ecospold_priority_process_value.setText(f"{current} / {update.process_total}")
            self.ecospold_priority_progress_bar.setRange(0, update.process_total)
            self.ecospold_priority_progress_bar.setValue(current)
        else:
            self.ecospold_priority_process_value.setText("0 / 0")
            self.ecospold_priority_progress_bar.setRange(0, stage_total)
            self.ecospold_priority_progress_bar.setValue(stage_current)

        self.ecospold_priority_current_process_value.setText(update.process_name or update.message)
        self.append_ecospold_priority_status(update.message)

    def run_inspect(self) -> None:
        if self._reduction_thread is not None:
            return
        database = self.database_edit.text().strip()
        if not database:
            QMessageBox.warning(self, "Missing input", "Select a database archive first.")
            return
        self.set_busy(True, "Inspecting database...")
        worker = InspectWorker(database, self.methods_edit.text().strip() or None)
        worker.finished.connect(self._handle_inspect_result)
        worker.failed.connect(self._handle_worker_failure)
        self._start_reduction_worker(worker, mode="inspect")

    @Slot(object)
    def _handle_inspect_result(self, result: object) -> None:
        if not isinstance(result, dict):
            return
        self._set_database_methods_hint(result)
        if result.get("database_methods_hint"):
            self.append_status(result["database_methods_hint"])
        self.append_status(json.dumps(result, indent=2, ensure_ascii=True))

    def run_create(self) -> None:
        if self._reduction_thread is not None:
            return
        database = self.database_edit.text().strip()
        output = self.output_edit.text().strip()
        if not database or not output:
            QMessageBox.warning(self, "Missing input", "Select a database archive and an output folder.")
            return

        self.status_box.clear()
        self.output_box.clear()
        self.reset_run_metrics()
        self.set_busy(True, "Creating reduced database...")
        try:
            tau = float(self.tau_edit.text())
        except ValueError:
            self.set_busy(False, "Create aborted.")
            QMessageBox.warning(self, "Invalid tau", "Tau must be a numeric value in (0, 1].")
            return

        worker = CreateWorker(
            database=database,
            methods=self.methods_edit.text().strip() or None,
            output=output,
            tau=tau,
            method_selection=self.selection_edit.text(),
            uncharacterised_policy=self.policy_combo.currentText(),
            strict_units=self.strict_units.isChecked(),
            tolerance=1e-12,
            allow_water_mass_volume_override=self.allow_water_mass_volume_override.isChecked(),
        )
        worker.progress.connect(self.update_create_progress)
        worker.finished.connect(self._handle_create_result)
        worker.failed.connect(self._handle_worker_failure)
        self._start_reduction_worker(worker, mode="create")

    def run_ecospold_create(self) -> None:
        if self._reduction_thread is not None:
            return
        database = self.ecospold_database_edit.text().strip()
        output = self.ecospold_output_edit.text().strip()
        if not database or not output:
            QMessageBox.warning(
                self,
                "Missing input",
                "Select an EcoSpold1 process archive or folder and an output folder.",
            )
            return

        self.ecospold_status_box.clear()
        self.ecospold_output_box.clear()
        self.reset_ecospold_metrics()
        self.set_ecospold_busy(True, "Creating reduced EcoSpold archive...")
        try:
            tau = float(self.ecospold_tau_edit.text())
        except ValueError:
            self.set_ecospold_busy(False, "EcoSpold reduction aborted.")
            QMessageBox.warning(self, "Invalid tau", "Tau must be a numeric value in (0, 1].")
            return

        worker = CreateWorker(
            database=database,
            methods=self.ecospold_methods_edit.text().strip() or None,
            output=output,
            tau=tau,
            method_selection=self.ecospold_selection_edit.text(),
            uncharacterised_policy=self.ecospold_policy_combo.currentText(),
            strict_units=self.ecospold_strict_units.isChecked(),
            tolerance=1e-12,
            allow_water_mass_volume_override=self.ecospold_allow_water_mass_volume_override.isChecked(),
        )
        worker.progress.connect(self.update_ecospold_progress)
        worker.finished.connect(self._handle_ecospold_create_result)
        worker.failed.connect(self._handle_worker_failure)
        self._start_reduction_worker(worker, mode="ecospold_create")

    def _parse_audit_tau_values(self, value: str) -> list[float]:
        raw_text = value.replace("\n", " ").replace(",", " ")
        tokens = [token.strip() for token in raw_text.split()]
        result: list[float] = []
        for token in tokens:
            if not token:
                continue
            result.append(float(token))
        if not result:
            raise ValueError("Provide at least one audit tau value.")
        return result

    def _parse_diagnostic_tau_values(self, value: str) -> list[float]:
        raw_text = value.replace("\n", " ").replace(",", " ")
        tokens = [token.strip() for token in raw_text.split() if token.strip()]
        if not tokens:
            raise ValueError("Provide at least one tau value.")
        return normalise_diagnostic_tau_values([float(token) for token in tokens])

    def update_diagnostic_progress(self, message: str, current: int, total: int) -> None:
        safe_total = max(total, 1)
        safe_current = min(max(current, 0), safe_total)
        self.diagnostic_stage_value.setText(f"{safe_current} / {safe_total}")
        self.diagnostic_progress_bar.setRange(0, safe_total)
        self.diagnostic_progress_bar.setValue(safe_current)
        self.append_diagnostic_status(message)

    def run_greedy_exact_diagnostic(self) -> None:
        if self._reduction_thread is not None:
            return
        database = self.diagnostic_database_edit.text().strip()
        output = self.diagnostic_output_edit.text().strip()
        process_query = self.diagnostic_process_edit.text().strip()
        if not database or not output:
            QMessageBox.warning(
                self,
                "Missing input",
                "Select a database archive and output folder before running the diagnostic.",
            )
            return
        try:
            tau_values = self._parse_diagnostic_tau_values(self.diagnostic_tau_edit.text())
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid tau values", str(exc))
            return

        self.diagnostic_status_box.clear()
        self.diagnostic_output_box.clear()
        self.reset_diagnostic_metrics()
        self.diagnostic_process_value.setText(process_query or "All processes")
        self.set_diagnostic_busy(True, "Running greedy vs exact diagnostic...")
        config = GreedyExactDiagnosticConfig(
            database=database,
            methods=self.diagnostic_methods_edit.text().strip() or None,
            output_dir=output,
            method_selection=self.diagnostic_selection_edit.text().strip() or "all",
            process_query=process_query,
            tau_values=tau_values,
            sign_mode=str(self.diagnostic_sign_mode_combo.currentData()),
            strict_units=self.diagnostic_strict_units.isChecked(),
            tolerance=1e-12,
            allow_water_mass_volume_override=self.diagnostic_allow_water_mass_volume_override.isChecked(),
        )
        worker = GreedyExactDiagnosticWorker(config)
        worker.progress.connect(self.update_diagnostic_progress)
        worker.finished.connect(self._handle_diagnostic_result)
        worker.failed.connect(self._handle_worker_failure)
        self._start_reduction_worker(worker, mode="diagnostic")

    def run_priority(self) -> None:
        if self._reduction_thread is not None:
            return
        database = self.priority_database_edit.text().strip()
        output = self.priority_output_edit.text().strip()
        if not database or not output:
            QMessageBox.warning(self, "Missing input", "Select a database archive and an output folder.")
            return

        self.priority_status_box.clear()
        self.priority_output_box.clear()
        self.reset_priority_metrics()
        self.set_priority_busy(True, "Generating flow-priority files...")
        try:
            audit_tau = self._parse_audit_tau_values(self.priority_audit_tau_edit.text())
        except ValueError as exc:
            self.set_priority_busy(False, "Flow-priority audit aborted.")
            QMessageBox.warning(self, "Invalid audit tau values", str(exc))
            return

        worker = PriorityWorker(
            database=database,
            methods=self.priority_methods_edit.text().strip() or None,
            output=output,
            method_selection=self.priority_selection_edit.text(),
            audit_tau=audit_tau,
            strict_units=self.priority_strict_units.isChecked(),
            tolerance=1e-12,
            allow_water_mass_volume_override=self.priority_allow_water_mass_volume_override.isChecked(),
        )
        worker.progress.connect(self.update_priority_progress)
        worker.finished.connect(self._handle_priority_result)
        worker.failed.connect(self._handle_worker_failure)
        self._start_reduction_worker(worker, mode="priority")

    def run_ambiguity_explorer(self) -> None:
        if self._reduction_thread is not None:
            return
        database = self.ambiguity_database_edit.text().strip()
        output = self.ambiguity_output_edit.text().strip()
        if not database or not output:
            QMessageBox.warning(
                self,
                "Missing input",
                "Select a database archive and an output folder.",
            )
            return

        self.ambiguity_status_box.clear()
        self.ambiguity_output_box.clear()
        self.reset_ambiguity_metrics()
        self.set_ambiguity_busy(True, "Exploring LCIA ambiguities...")
        try:
            tolerance = float(self.ambiguity_tolerance_edit.text())
        except ValueError:
            self.set_ambiguity_busy(False, "Ambiguity exploration aborted.")
            QMessageBox.warning(self, "Invalid tolerance", "Tolerance must be a numeric value.")
            return

        worker = AmbiguityExplorerWorker(
            AmbiguityExploreConfig(
                database=database,
                methods=self.ambiguity_methods_edit.text().strip() or None,
                output_dir=output,
                method_selection=self.ambiguity_selection_edit.text().strip() or "all",
                strict_units=self.ambiguity_strict_units.isChecked(),
                tolerance=tolerance,
                allow_water_mass_volume_override=self.ambiguity_allow_water_mass_volume_override.isChecked(),
            )
        )
        worker.progress.connect(self.update_ambiguity_progress)
        worker.finished.connect(self._handle_ambiguity_result)
        worker.failed.connect(self._handle_worker_failure)
        self._start_reduction_worker(worker, mode="ambiguity")

    def run_ecospold_ambiguity_explorer(self) -> None:
        if self._reduction_thread is not None:
            return
        database = self.ecospold_ambiguity_database_edit.text().strip()
        output = self.ecospold_ambiguity_output_edit.text().strip()
        if not database or not output:
            QMessageBox.warning(
                self,
                "Missing input",
                "Select an EcoSpold1 process archive or folder and an output folder.",
            )
            return

        self.ecospold_ambiguity_status_box.clear()
        self.ecospold_ambiguity_output_box.clear()
        self.reset_ecospold_ambiguity_metrics()
        self.set_ecospold_ambiguity_busy(True, "Exploring EcoSpold1 LCIA ambiguities...")
        try:
            tolerance = float(self.ecospold_ambiguity_tolerance_edit.text())
        except ValueError:
            self.set_ecospold_ambiguity_busy(False, "EcoSpold1 ambiguity exploration aborted.")
            QMessageBox.warning(self, "Invalid tolerance", "Tolerance must be a numeric value.")
            return

        worker = AmbiguityExplorerWorker(
            AmbiguityExploreConfig(
                database=database,
                methods=self.ecospold_ambiguity_methods_edit.text().strip() or None,
                output_dir=output,
                method_selection=self.ecospold_ambiguity_selection_edit.text().strip() or "all",
                strict_units=self.ecospold_ambiguity_strict_units.isChecked(),
                tolerance=tolerance,
                allow_water_mass_volume_override=self.ecospold_ambiguity_allow_water_mass_volume_override.isChecked(),
            )
        )
        worker.progress.connect(self.update_ecospold_ambiguity_progress)
        worker.finished.connect(self._handle_ecospold_ambiguity_result)
        worker.failed.connect(self._handle_worker_failure)
        self._start_reduction_worker(worker, mode="ecospold_ambiguity")

    def run_ecospold_priority(self) -> None:
        if self._reduction_thread is not None:
            return
        database = self.ecospold_priority_database_edit.text().strip()
        output = self.ecospold_priority_output_edit.text().strip()
        if not database or not output:
            QMessageBox.warning(
                self,
                "Missing input",
                "Select an EcoSpold1 process archive or folder and an output folder.",
            )
            return

        self.ecospold_priority_status_box.clear()
        self.ecospold_priority_output_box.clear()
        self.reset_ecospold_priority_metrics()
        self.set_ecospold_priority_busy(True, "Generating EcoSpold flow-priority files...")
        try:
            audit_tau = self._parse_audit_tau_values(self.ecospold_priority_audit_tau_edit.text())
        except ValueError as exc:
            self.set_ecospold_priority_busy(False, "EcoSpold flow-priority audit aborted.")
            QMessageBox.warning(self, "Invalid audit tau values", str(exc))
            return

        worker = PriorityWorker(
            database=database,
            methods=self.ecospold_priority_methods_edit.text().strip() or None,
            output=output,
            method_selection=self.ecospold_priority_selection_edit.text(),
            audit_tau=audit_tau,
            strict_units=self.ecospold_priority_strict_units.isChecked(),
            tolerance=1e-12,
            allow_water_mass_volume_override=self.ecospold_priority_allow_water_mass_volume_override.isChecked(),
        )
        worker.progress.connect(self.update_ecospold_priority_progress)
        worker.finished.connect(self._handle_ecospold_priority_result)
        worker.failed.connect(self._handle_worker_failure)
        self._start_reduction_worker(worker, mode="ecospold_priority")

    @Slot(object)
    def _handle_create_result(self, result: object) -> None:
        output_text: list[str] = []
        if hasattr(result, "output_zip"):
            output_text.extend(
                [
                    result.output_zip,
                    result.run_summary_json,
                    result.reduction_debug_ndjson,
                ]
            )
            self.set_output_paths(output_text)
            self.append_status(json.dumps(result.summary, indent=2, ensure_ascii=True))

    @Slot(object)
    def _handle_ecospold_create_result(self, result: object) -> None:
        output_text: list[str] = []
        if hasattr(result, "output_zip"):
            output_text.extend(
                [
                    result.output_zip,
                    result.run_summary_json,
                    result.reduction_debug_ndjson,
                ]
            )
            self.set_ecospold_output_paths(output_text)
            self.append_ecospold_status(json.dumps(result.summary, indent=2, ensure_ascii=True))

    @Slot(object)
    def _handle_priority_result(self, result: object) -> None:
        output_text: list[str] = []
        if hasattr(result, "flow_priority_csv"):
            output_text.extend(
                [
                    result.flow_priority_csv,
                    result.flow_priority_metadata_json,
                ]
            )
            self.set_priority_output_paths(output_text)
            self.append_priority_status(json.dumps(result.metadata, indent=2, ensure_ascii=True))

    @Slot(object)
    def _handle_ambiguity_result(self, result: object) -> None:
        output_text: list[str] = []
        if hasattr(result, "cf_ambiguities_csv"):
            output_text.extend(
                [
                    result.cf_ambiguities_csv,
                    result.cf_ambiguity_metadata_json,
                ]
            )
            self.set_ambiguity_output_paths(output_text)
            self.append_ambiguity_status(json.dumps(result.metadata, indent=2, ensure_ascii=True))

    @Slot(object)
    def _handle_ecospold_ambiguity_result(self, result: object) -> None:
        output_text: list[str] = []
        if hasattr(result, "cf_ambiguities_csv"):
            output_text.extend(
                [
                    result.cf_ambiguities_csv,
                    result.cf_ambiguity_metadata_json,
                ]
            )
            self.set_ecospold_ambiguity_output_paths(output_text)
            self.append_ecospold_ambiguity_status(json.dumps(result.metadata, indent=2, ensure_ascii=True))

    @Slot(object)
    def _handle_ecospold_priority_result(self, result: object) -> None:
        output_text: list[str] = []
        if hasattr(result, "flow_priority_csv"):
            output_text.extend(
                [
                    result.flow_priority_csv,
                    result.flow_priority_metadata_json,
                ]
            )
            self.set_ecospold_priority_output_paths(output_text)
            self.append_ecospold_priority_status(json.dumps(result.metadata, indent=2, ensure_ascii=True))

    @Slot(object)
    def _handle_diagnostic_result(self, result: object) -> None:
        if not hasattr(result, "metadata"):
            return
        metadata = result.metadata
        summary = metadata.get("summary") or {}
        selected_process = metadata.get("selected_process") or {}
        if metadata.get("batch_mode"):
            self.diagnostic_process_value.setText(f"All processes ({summary.get('n_processes', 0)})")
            self.diagnostic_protected_value.setText("-")
        else:
            self.diagnostic_process_value.setText(
                f"{selected_process['process_name']} [{selected_process['process_id']}]"
            )
            self.diagnostic_protected_value.setText(str(selected_process["protected_exchanges"]))
        self.diagnostic_greedy_coverage_value.setText(summary.get("tau_values_text", "-"))
        self.diagnostic_exact_coverage_value.setText(
            (
                f"{summary.get('n_exact_clones_written', 0)} / {summary.get('n_tau_values', 0)}"
                if not metadata.get("batch_mode")
                else str(sum(int(item.get("exact_solved_processes", 0)) for item in metadata.get("aggregate_by_tau") or []))
            )
        )
        greedy_ok = bool(summary.get("all_greedy_certificates_pass"))
        exact_optimal = bool(summary.get("all_exact_tau_optimal"))
        exact_valid = bool(summary.get("all_exact_results_valid"))
        self.diagnostic_certificate_value.setText(
            f"greedy={'pass' if greedy_ok else 'fail'} | exact_valid={'yes' if exact_valid else 'no'} | exact_all_tau={'yes' if exact_optimal else 'no'}"
        )
        tau_results = metadata.get("tau_results") or []
        if not tau_results and metadata.get("process_results"):
            tau_results = [
                tau_row
                for process_row in metadata.get("process_results") or []
                for tau_row in process_row.get("tau_results") or []
            ]
        runtime_total = sum(float(row.get("runtime_seconds") or 0.0) for row in tau_results)
        self.diagnostic_runtime_value.setText(f"{runtime_total:.3f}s")
        self.set_diagnostic_output_paths(
            [path for path in [result.diagnostic_zip, result.metadata_json, result.summary_csv, result.debug_csv] if path]
        )
        self.append_diagnostic_status(json.dumps(metadata, indent=2, ensure_ascii=True))

    def _curve_cache_key(self, source_path: str) -> str:
        path = Path(source_path)
        stat = path.stat()
        return f"{path.resolve()}::{stat.st_mtime_ns}::{stat.st_size}"

    def _group_by_id(self, group_id: str) -> DatabaseReductionGroup | None:
        for group in self.curve_groups:
            if group.id == group_id:
                return group
        return None

    def _run_by_id(self, run_id: str) -> tuple[DatabaseReductionGroup | None, TauReductionRun | None]:
        for group in self.curve_groups:
            for run in group.runs:
                if run.id == run_id:
                    return group, run
        return None, None

    def add_curve_group(self) -> None:
        proposed_name = self.curve_group_name_edit.text().strip() or f"Database {len(self.curve_groups) + 1}"
        name = self._unique_group_name(proposed_name)
        self.curve_groups.append(DatabaseReductionGroup(id=uuid.uuid4().hex, name=name))
        self.curve_group_name_edit.clear()
        self._refresh_curve_views()

    def _unique_group_name(self, proposed_name: str) -> str:
        existing = {group.name for group in self.curve_groups}
        if proposed_name not in existing:
            return proposed_name
        index = 2
        while f"{proposed_name} {index}" in existing:
            index += 1
        return f"{proposed_name} {index}"

    def pick_curve_run_files(self, group_id: str) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select reduction artefacts",
            "",
            "Reduction outputs (*.json *.csv *.zip *.zolca *.pdf);;All files (*)",
        )
        for path in paths:
            self._queue_curve_run(group_id, path)

    def pick_curve_run_folder(self, group_id: str) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select reduction output folder")
        if path:
            self._queue_curve_run(group_id, path)

    def _queue_curve_run(self, group_id: str, source_path: str) -> None:
        group = self._group_by_id(group_id)
        if group is None:
            return
        source = Path(source_path)
        run_id = uuid.uuid4().hex
        try:
            cache_key = self._curve_cache_key(source_path)
        except OSError as exc:
            QMessageBox.warning(self, "Input unavailable", str(exc))
            return

        if cache_key in self._curve_cache:
            cached = clone_run(self._curve_cache[cache_key])
            cached.id = run_id
            cached.sourceFileName = source.name
            cached.sourcePath = str(source)
            cached.state = "ready"
            cached.statusMessage = "Loaded from cache."
            group.runs.append(cached)
            self._refresh_curve_views()
            return

        run = TauReductionRun(
            id=run_id,
            sourceFileName=source.name,
            tau=None,
            elementaryBefore=None,
            elementaryAfter=None,
            elementaryRemoved=None,
            retainedPercent=None,
            removedPercent=None,
            validationStatus="unknown",
            sourcePath=str(source),
            runDirectory=str(source if source.is_dir() else source.parent),
            state="queued",
            statusMessage="Queued for metadata extraction.",
        )
        run.refresh_warnings()
        group.runs.append(run)
        self._curve_queue.append(run_id)
        self._refresh_curve_views()
        self._start_next_curve_job()

    def remove_curve_run(self, group_id: str, run_id: str) -> None:
        group = self._group_by_id(group_id)
        if group is None:
            return
        self._curve_removed_run_ids.add(run_id)
        self._curve_queue = [queued_id for queued_id in self._curve_queue if queued_id != run_id]
        if self._curve_active_run_id == run_id and self._curve_worker is not None:
            self._curve_worker.cancel()
        group.runs = [run for run in group.runs if run.id != run_id]
        self._refresh_curve_views()
        self._start_next_curve_job()

    def remove_curve_group(self, group_id: str) -> None:
        group = self._group_by_id(group_id)
        if group is None:
            return
        for run in group.runs:
            self._curve_removed_run_ids.add(run.id)
            self._curve_queue = [queued_id for queued_id in self._curve_queue if queued_id != run.id]
            if self._curve_active_run_id == run.id and self._curve_worker is not None:
                self._curve_worker.cancel()
        self.curve_groups = [item for item in self.curve_groups if item.id != group_id]
        self._refresh_curve_views()
        self._start_next_curve_job()

    def _start_next_curve_job(self) -> None:
        if self._curve_thread is not None:
            return
        while self._curve_queue:
            run_id = self._curve_queue.pop(0)
            group, run = self._run_by_id(run_id)
            if group is None or run is None:
                continue
            run.state = "processing"
            run.statusMessage = "Reading reduction artefacts."
            run.refresh_warnings()
            worker = CurveMetadataWorker(run.id, run.sourcePath)
            thread = QThread(self)
            worker.moveToThread(thread)
            thread.started.connect(worker.run)
            worker.done.connect(thread.quit)
            worker.done.connect(worker.deleteLater)
            thread.finished.connect(thread.deleteLater)
            thread.finished.connect(self._on_curve_worker_finished)
            worker.progress.connect(self._handle_curve_progress)
            worker.finished.connect(self._handle_curve_finished)
            worker.failed.connect(self._handle_curve_failed)
            worker.cancelled.connect(self._handle_curve_cancelled)
            self._curve_thread = thread
            self._curve_worker = worker
            self._curve_active_run_id = run.id
            thread.start()
            self._refresh_curve_views()
            return
        self.curve_status_label.setText("No reduction-curve uploads in progress.")
        self.curve_progress_bar.setRange(0, 1)
        self.curve_progress_bar.setValue(0)

    @Slot(str, int, int)
    def _handle_curve_progress(self, message: str, current: int, total: int) -> None:
        self.curve_status_label.setText(message)
        total_value = max(total, 1)
        current_value = min(max(current, 0), total_value)
        self.curve_progress_bar.setRange(0, total_value)
        self.curve_progress_bar.setValue(current_value)
        _, run = self._run_by_id(self._curve_active_run_id or "")
        if run is not None:
            run.statusMessage = message
        self._refresh_curve_views()

    @Slot(str, object)
    def _handle_curve_finished(self, run_id: str, result: object) -> None:
        group, existing = self._run_by_id(run_id)
        if group is None or existing is None or run_id in self._curve_removed_run_ids:
            return
        if not isinstance(result, TauReductionRun):
            return
        result.id = run_id
        result.sourcePath = existing.sourcePath
        result.sourceFileName = existing.sourceFileName
        result.state = "ready"
        result.statusMessage = "Metadata extracted."
        result.groupWarnings = list(existing.groupWarnings)
        result.refresh_warnings()
        for index, run in enumerate(group.runs):
            if run.id == run_id:
                group.runs[index] = result
                break
        try:
            cache_key = self._curve_cache_key(result.sourcePath)
            self._curve_cache[cache_key] = clone_run(result)
        except OSError:
            pass
        self._refresh_curve_views()

    @Slot(str, str)
    def _handle_curve_failed(self, run_id: str, message: str) -> None:
        group, run = self._run_by_id(run_id)
        if group is None or run is None or run_id in self._curve_removed_run_ids:
            return
        run.state = "failed"
        run.statusMessage = message
        run.sourceWarnings = [message]
        run.refresh_warnings()
        self._refresh_curve_views()

    @Slot(str)
    def _handle_curve_cancelled(self, run_id: str) -> None:
        group, run = self._run_by_id(run_id)
        if group is None or run is None:
            return
        run.state = "cancelled"
        run.statusMessage = "Metadata extraction cancelled."
        run.sourceWarnings = [run.statusMessage]
        run.refresh_warnings()
        self._refresh_curve_views()

    @Slot()
    def _on_curve_worker_finished(self) -> None:
        self._curve_thread = None
        self._curve_worker = None
        self._curve_active_run_id = None
        self.curve_progress_bar.setRange(0, 1)
        self.curve_progress_bar.setValue(0)
        self._refresh_curve_views()
        self._start_next_curve_job()

    def _apply_curve_group_warnings(self) -> None:
        for group in self.curve_groups:
            warnings_by_run = group_warnings(group.runs)
            for run in group.runs:
                run.groupWarnings = warnings_by_run.get(run.id, [])
                run.refresh_warnings()

    def _refresh_curve_views(self) -> None:
        self._apply_curve_group_warnings()
        self._render_curve_groups()
        self._update_curve_charts()
        if self.curve_export_button is not None:
            self.curve_export_button.setEnabled(any(group.runs for group in self.curve_groups))

    def _render_curve_groups(self) -> None:
        while self.curve_groups_layout.count():
            item = self.curve_groups_layout.takeAt(0)
            widget = item.widget()
            child_layout = item.layout()
            if widget is not None:
                widget.deleteLater()
            elif child_layout is not None:
                child_layout.deleteLater()

        if not self.curve_groups:
            empty = QLabel("Add a database group, then attach completed reduction outputs for multiple tau values.")
            empty.setObjectName("muted")
            empty.setWordWrap(True)
            self.curve_groups_layout.addWidget(empty)
            self.curve_groups_layout.addStretch(1)
            return

        for group in self.curve_groups:
            self.curve_groups_layout.addWidget(self._build_curve_group_card(group))
        self.curve_groups_layout.addStretch(1)

    def _build_curve_group_card(self, group: DatabaseReductionGroup) -> QWidget:
        card = QFrame()
        card.setObjectName("groupCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(10)

        header = QHBoxLayout()
        title = QLabel(group.name)
        title.setObjectName("groupTitle")
        header.addWidget(title)
        summary = QLabel(f"{len(group.runs)} run(s)")
        summary.setObjectName("muted")
        header.addWidget(summary)
        header.addStretch(1)

        add_files_button = QPushButton("Add files")
        add_files_button.setObjectName("secondary")
        add_files_button.clicked.connect(lambda: self.pick_curve_run_files(group.id))
        add_folder_button = QPushButton("Add folder")
        add_folder_button.setObjectName("secondary")
        add_folder_button.clicked.connect(lambda: self.pick_curve_run_folder(group.id))
        remove_group_button = QPushButton("Remove database")
        remove_group_button.setObjectName("danger")
        remove_group_button.clicked.connect(lambda: self.remove_curve_group(group.id))
        header.addWidget(add_files_button)
        header.addWidget(add_folder_button)
        header.addWidget(remove_group_button)
        layout.addLayout(header)

        unique_group_warnings = sorted({warning for run in group.runs for warning in run.groupWarnings})
        if unique_group_warnings:
            warning_label = QLabel("Warnings: " + " | ".join(unique_group_warnings))
            warning_label.setObjectName("warningText")
            warning_label.setWordWrap(True)
            layout.addWidget(warning_label)

        table = QTableWidget(len(group.runs), 8)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setSelectionMode(QAbstractItemView.SingleSelection)
        table.verticalHeader().setVisible(False)
        table.setAlternatingRowColors(False)
        table.setHorizontalHeaderLabels(
            [
                "Tau",
                "Elementary before",
                "Elementary after",
                "Retained %",
                "Removed %",
                "Validation",
                "Source file",
                "Remove",
            ]
        )
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(6, QHeaderView.Stretch)
        table.horizontalHeader().setSectionResizeMode(7, QHeaderView.ResizeToContents)

        for row_index, run in enumerate(_sort_runs(group.runs)):
            self._set_curve_table_item(table, row_index, 0, "-" if run.tau is None else f"{run.tau:.4g}", run)
            self._set_curve_table_item(table, row_index, 1, _format_int(run.elementaryBefore), run)
            self._set_curve_table_item(table, row_index, 2, _format_int(run.elementaryAfter), run)
            self._set_curve_table_item(table, row_index, 3, _format_percent(run.retainedPercent), run)
            self._set_curve_table_item(table, row_index, 4, _format_percent(run.removedPercent), run)
            self._set_curve_table_item(table, row_index, 5, self._display_validation_status(run), run, status_column=True)
            self._set_curve_table_item(table, row_index, 6, run.sourceFileName, run)
            remove_button = QPushButton("Remove")
            remove_button.setObjectName("danger")
            remove_button.clicked.connect(lambda _checked=False, gid=group.id, rid=run.id: self.remove_curve_run(gid, rid))
            table.setCellWidget(row_index, 7, remove_button)

        layout.addWidget(table)
        return card

    def _set_curve_table_item(
        self,
        table: QTableWidget,
        row: int,
        column: int,
        text: str,
        run: TauReductionRun,
        *,
        status_column: bool = False,
    ) -> None:
        item = QTableWidgetItem(text)
        item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
        tooltip_parts = []
        if run.statusMessage:
            tooltip_parts.append(run.statusMessage)
        if run.inputDatabaseName:
            tooltip_parts.append(f"Input database: {run.inputDatabaseName}")
        if run.inputDatabaseHash:
            tooltip_parts.append(f"Input hash: {run.inputDatabaseHash}")
        tooltip_parts.extend(run.warnings)
        if tooltip_parts:
            item.setToolTip("\n".join(dict.fromkeys(tooltip_parts)))
        if status_column:
            item.setForeground(self._status_brush(run))
        table.setItem(row, column, item)

    def _display_validation_status(self, run: TauReductionRun) -> str:
        if run.state in {"queued", "processing", "cancelled", "failed"}:
            return run.state
        return run.validationStatus

    def _status_brush(self, run: TauReductionRun):
        status = self._display_validation_status(run)
        color = "#475569"
        if status == "pass":
            color = "#1f5f4a"
        elif status in {"fail", "failed"}:
            color = "#8c3a2f"
        elif status == "processing":
            color = "#234b63"
        elif status == "queued":
            color = "#7c5d21"
        return QBrush(QColor(color))

    def _update_curve_charts(self) -> None:
        if not QT_CHARTS_AVAILABLE:
            return
        assert isinstance(self.curve_removed_chart, QChartView)
        assert isinstance(self.curve_retained_chart, QChartView)
        self._populate_curve_chart(
            self.curve_removed_chart,
            title="Reduction curve",
            y_label="Removed %",
            value_attr="removedPercent",
        )
        self._populate_curve_chart(
            self.curve_retained_chart,
            title="Retained curve",
            y_label="Retained %",
            value_attr="retainedPercent",
        )

    def _populate_curve_chart(self, view: QChartView, *, title: str, y_label: str, value_attr: str) -> None:
        chart = QChart()
        chart.setTitle(title)
        chart.legend().setVisible(True)
        chart.legend().setAlignment(Qt.AlignBottom)

        x_values: list[float] = []
        palette = ["#234b63", "#2f6f5e", "#8c3a2f", "#56657a", "#7f6a2e", "#4d3f6b"]
        for index, group in enumerate(self.curve_groups):
            valid_runs = [
                run
                for run in _sort_runs(group.runs)
                if curve_point_is_valid(run) and getattr(run, value_attr) is not None
            ]
            if not valid_runs:
                continue
            series = QLineSeries()
            series.setName(group.name)
            series.setPointsVisible(True)
            pen = QPen(QColor(palette[index % len(palette)]))
            pen.setWidth(2)
            series.setPen(pen)
            for run in valid_runs:
                x_values.append(float(run.tau))
                series.append(float(run.tau), float(getattr(run, value_attr)))
            chart.addSeries(series)

        axis_x = QValueAxis()
        axis_x.setTitleText("Tau")
        axis_x.setLabelFormat("%.3g")
        if x_values:
            xmin = min(x_values)
            xmax = max(x_values)
            if xmin == xmax:
                padding = 0.05 if xmin == 0 else abs(xmin) * 0.05
                axis_x.setRange(max(0.0, xmin - padding), min(1.0, xmax + padding) if xmax <= 1.0 else xmax + padding)
            else:
                padding = max((xmax - xmin) * 0.05, 0.01)
                axis_x.setRange(max(0.0, xmin - padding), min(1.0, xmax + padding) if xmax <= 1.0 else xmax + padding)
        else:
            axis_x.setRange(0.0, 1.0)
        chart.addAxis(axis_x, Qt.AlignBottom)

        axis_y = QValueAxis()
        axis_y.setTitleText(y_label)
        axis_y.setLabelFormat("%.1f")
        axis_y.setRange(0.0, 100.0)
        chart.addAxis(axis_y, Qt.AlignLeft)

        for series in chart.series():
            series.attachAxis(axis_x)
            series.attachAxis(axis_y)
        view.setChart(chart)

    def export_curve_csv(self) -> None:
        rows = export_curve_rows((group.name, group.runs) for group in self.curve_groups)
        if not rows:
            QMessageBox.information(self, "No data", "There are no reduction-curve runs to export.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export reduction-curve CSV",
            "reduction_curves.csv",
            "CSV files (*.csv)",
        )
        if not path:
            return
        fieldnames = list(rows[0].keys())
        with Path(path).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        self.curve_status_label.setText(f"Curve CSV exported to {path}")


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    _place_main_window(window, app)
    window.show()
    return app.exec()


def _place_main_window(window: MainWindow, app: QApplication) -> None:
    screen = app.primaryScreen()
    if screen is None:
        window.resize(1320, 920)
        return
    available = screen.availableGeometry()
    width = min(1320, available.width(), max(720, available.width() - 80))
    height = min(920, available.height(), max(560, available.height() - 80))
    x = available.x() + max(0, (available.width() - width) // 2)
    y = available.y() + max(0, (available.height() - height) // 2)
    window.setGeometry(x, y, width, height)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
