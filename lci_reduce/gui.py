"""PySide6 GUI for lci_reduce."""

from __future__ import annotations

import csv
import json
import sys
import threading
import uuid
from pathlib import Path

from PySide6.QtCore import QObject, QThread, Qt, Signal, Slot
from PySide6.QtGui import QBrush, QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
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

from .cf_resolution import CFAmbiguityContext, CFPromptResult, candidate_display_text
from .cli import CLI_GUIDE_SECTIONS, create_command, inspect_command, priority_command
from .errors import RunCancelledError
from .models import CreateProgressUpdate, DatabaseReductionGroup, TauReductionRun
from .priority_analyser_gui import PriorityAnalyserPanel
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


class CFAmbiguityDialog(QDialog):
    def __init__(self, parent, context: CFAmbiguityContext, candidate_texts: list[str]) -> None:
        super().__init__(parent)
        self.setWindowTitle("Resolve CF ambiguity")
        self.prompt_result = CFPromptResult(action="cancel_run")

        layout = QVBoxLayout(self)
        summary = QLabel(
            "\n".join(
                [
                    f"Process: {context.process_name or context.process_id or '-'}",
                    f"Exchange/Flow: {context.flow_name or context.flow_id or '-'} ({context.exchange_id or '-'})",
                    f"LCIA Category: {context.category_name or context.category_id or '-'}",
                    f"Differing fields: {', '.join(context.differing_fields) or 'none'}",
                ]
            )
        )
        summary.setWordWrap(True)
        layout.addWidget(summary)

        self.choice_list = QListWidget()
        for index, text in enumerate(candidate_texts, start=1):
            self.choice_list.addItem(f"Candidate {index}\n{text}")
        if candidate_texts:
            self.choice_list.setCurrentRow(0)
        self.choice_list.setMinimumHeight(320)
        layout.addWidget(self.choice_list)

        buttons = QHBoxLayout()
        select_button = QPushButton("Select")
        select_button.clicked.connect(self._select_choice)
        skip_button = QPushButton("Skip or fail")
        skip_button.clicked.connect(self._skip_fail)
        cancel_button = QPushButton("Cancel run")
        cancel_button.setObjectName("danger")
        cancel_button.clicked.connect(self._cancel_run)
        buttons.addWidget(select_button)
        buttons.addWidget(skip_button)
        buttons.addWidget(cancel_button)
        layout.addLayout(buttons)

    def _select_choice(self) -> None:
        row = self.choice_list.currentRow()
        if row < 0:
            QMessageBox.warning(self, "Selection required", "Select a candidate before continuing.")
            return
        self.prompt_result = CFPromptResult(action="select", candidate_index=row)
        self.accept()

    def _skip_fail(self) -> None:
        self.prompt_result = CFPromptResult(action="skip_fail")
        self.done(0)

    def _cancel_run(self) -> None:
        self.prompt_result = CFPromptResult(action="cancel_run")
        self.done(0)


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
    prompt_requested = Signal(object, object)
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
        cf_resolution_file: str | None,
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
        self.cf_resolution_file = cf_resolution_file
        self._prompt_lock = threading.Lock()
        self._prompt_event: threading.Event | None = None
        self._prompt_result: CFPromptResult | None = None

    def deliver_prompt_result(self, result: CFPromptResult) -> None:
        with self._prompt_lock:
            self._prompt_result = result
            event = self._prompt_event
        if event is not None:
            event.set()

    def _prompt_cf(self, context: CFAmbiguityContext, candidates) -> CFPromptResult:
        event = threading.Event()
        with self._prompt_lock:
            self._prompt_event = event
            self._prompt_result = CFPromptResult(action="cancel_run")
        self.prompt_requested.emit(context, list(candidates))
        event.wait()
        with self._prompt_lock:
            result = self._prompt_result or CFPromptResult(action="cancel_run")
            self._prompt_event = None
            self._prompt_result = None
        return result

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
                cf_resolution_file=self.cf_resolution_file,
                cf_prompt=self._prompt_cf,
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
    prompt_requested = Signal(object, object)
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
        cf_resolution_file: str | None,
    ) -> None:
        super().__init__()
        self.database = database
        self.methods = methods
        self.output = output
        self.method_selection = method_selection
        self.audit_tau = audit_tau
        self.strict_units = strict_units
        self.tolerance = tolerance
        self.cf_resolution_file = cf_resolution_file
        self._prompt_lock = threading.Lock()
        self._prompt_event: threading.Event | None = None
        self._prompt_result: CFPromptResult | None = None

    def deliver_prompt_result(self, result: CFPromptResult) -> None:
        with self._prompt_lock:
            self._prompt_result = result
            event = self._prompt_event
        if event is not None:
            event.set()

    def _prompt_cf(self, context: CFAmbiguityContext, candidates) -> CFPromptResult:
        event = threading.Event()
        with self._prompt_lock:
            self._prompt_event = event
            self._prompt_result = CFPromptResult(action="cancel_run")
        self.prompt_requested.emit(context, list(candidates))
        event.wait()
        with self._prompt_lock:
            result = self._prompt_result or CFPromptResult(action="cancel_run")
            self._prompt_event = None
            self._prompt_result = None
        return result

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
                cf_resolution_file=self.cf_resolution_file,
                cf_prompt=self._prompt_cf,
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


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("lci_reduce")

        self.database_edit = QLineEdit()
        self.methods_edit = QLineEdit()
        self.output_edit = QLineEdit()
        self.cf_choices_edit = QLineEdit()
        self.tau_edit = QLineEdit("0.95")
        self.selection_edit = QLineEdit("all")
        self.policy_combo = QComboBox()
        self.policy_combo.addItems(["keep", "drop", "fail"])
        self.policy_combo.setCurrentText("drop")
        self.strict_units = QCheckBox("Enforce strict unit compatibility")
        self.strict_units.setChecked(True)
        self.priority_database_edit = QLineEdit()
        self.priority_methods_edit = QLineEdit()
        self.priority_output_edit = QLineEdit()
        self.priority_cf_choices_edit = QLineEdit()
        self.priority_selection_edit = QLineEdit("all")
        self.priority_audit_tau_edit = QLineEdit("0.95, 0.99")
        self.priority_strict_units = QCheckBox("Enforce strict unit compatibility")
        self.priority_strict_units.setChecked(True)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.priority_progress_bar = QProgressBar()
        self.priority_progress_bar.setRange(0, 1)
        self.priority_progress_bar.setValue(0)
        self.priority_progress_bar.setTextVisible(False)

        self.database_methods_label = QLabel("Select a database archive to check for embedded LCIA methods.")
        self.database_methods_label.setObjectName("muted")
        self.database_methods_label.setWordWrap(True)
        self.priority_database_methods_label = QLabel("Select a database archive to check for embedded LCIA methods.")
        self.priority_database_methods_label.setObjectName("muted")
        self.priority_database_methods_label.setWordWrap(True)
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

        self.stage_value = QLabel("Idle")
        self.process_value = QLabel("0 / 0")
        self.exchange_value = QLabel("0 / 0")
        self.current_process_value = QLabel("Ready")
        self.current_process_value.setWordWrap(True)
        self.priority_stage_value = QLabel("Idle")
        self.priority_process_value = QLabel("0 / 0")
        self.priority_current_process_value = QLabel("Ready")
        self.priority_current_process_value.setWordWrap(True)

        self.inspect_button: QPushButton | None = None
        self.create_button: QPushButton | None = None
        self.priority_button: QPushButton | None = None
        self.use_database_methods_button: QPushButton | None = None
        self.priority_use_database_methods_button: QPushButton | None = None
        self.database_has_impact_methods = False
        self.priority_database_has_impact_methods = False

        self._reduction_thread: QThread | None = None
        self._reduction_worker: QObject | None = None
        self._reduction_mode = ""

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

        self._build()
        self._apply_style()
        self._refresh_curve_views()

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QWidget {
                background: #f3f5f7;
                color: #18222d;
                font-family: "Avenir Next", "Helvetica Neue", "Segoe UI";
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
        tabs.addTab(self._build_reduction_tab(), "Reduction")
        tabs.addTab(self._build_priority_tab(), "Flow priority")
        tabs.addTab(PriorityAnalyserPanel(), "Priority analyser")
        tabs.addTab(self._build_curves_tab(), "Reduction curves")
        tabs.addTab(self._build_cli_tab(), "CLI info")
        root.addWidget(tabs)

        self.setCentralWidget(central)

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

        load_cf_choices_button = QPushButton("Load choices")
        load_cf_choices_button.setObjectName("secondary")
        load_cf_choices_button.clicked.connect(self.pick_cf_choices_load)
        save_cf_choices_button = QPushButton("Save choices")
        save_cf_choices_button.setObjectName("secondary")
        save_cf_choices_button.clicked.connect(self.pick_cf_choices_save)
        inputs_form.addRow(
            "CF choices CSV",
            self._picker_row(
                self.cf_choices_edit,
                "Browse",
                self.pick_cf_choices_load,
                extra_buttons=[load_cf_choices_button, save_cf_choices_button],
            ),
        )

        settings_group = QGroupBox("Reduction settings")
        settings_form = QFormLayout(settings_group)
        settings_form.setSpacing(10)
        settings_form.addRow("Tau", self.tau_edit)
        settings_form.addRow("Method selection", self.selection_edit)
        settings_form.addRow("Uncharacterised policy", self.policy_combo)
        settings_form.addRow("", self.strict_units)

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
        inputs_form.addRow(
            "CF choices CSV",
            self._picker_row(
                self.priority_cf_choices_edit,
                "Browse",
                self.pick_priority_cf_choices_load,
            ),
        )

        settings_group = QGroupBox("Audit settings")
        settings_form = QFormLayout(settings_group)
        settings_form.setSpacing(10)
        settings_form.addRow("Method selection", self.priority_selection_edit)
        settings_form.addRow("Audit tau values", self.priority_audit_tau_edit)
        settings_form.addRow("", self.priority_strict_units)

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

    def _build_cli_tab(self) -> QWidget:
        tab = QWidget()
        root = QVBoxLayout(tab)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(12)

        root.addWidget(
            self._make_section_header(
                "CLI info",
                "These are copy-ready commands for the same backend used by the GUI. Follow the workflow order below when you want reproducible scripted runs.",
            )
        )

        intro = QFrame()
        intro.setObjectName("panel")
        intro_layout = QVBoxLayout(intro)
        intro_layout.setContentsMargins(14, 12, 14, 12)
        intro_title = QLabel("Recommended workflow")
        intro_title.setObjectName("sectionTitle")
        intro_title.setStyleSheet("font-size: 16px;")
        intro_body = QLabel(
            "1. Run `inspect` first.\n"
            "2. Use `create` when you need a lite JSON-LD database ZIP.\n"
            "3. Use `priority` when you need the LCIA-critical flow sidecars only.\n"
            "4. Use `analyse-priority` to screen an already generated `lcia_flow_priority.csv`.\n"
            "5. Use repeated `--select-flow-name` options when flow names contain commas."
        )
        intro_body.setObjectName("muted")
        intro_body.setWordWrap(True)
        intro_layout.addWidget(intro_title)
        intro_layout.addWidget(intro_body)
        root.addWidget(intro)

        safety = QFrame()
        safety.setObjectName("panel")
        safety_layout = QVBoxLayout(safety)
        safety_layout.setContentsMargins(14, 12, 14, 12)
        safety_title = QLabel("Safety and output expectations")
        safety_title.setObjectName("sectionTitle")
        safety_title.setStyleSheet("font-size: 16px;")
        safety_body = QLabel(
            "The tool is ZIP-only. It does not use openLCA IPC, does not connect to openLCA, and does not edit a live openLCA database.\n"
            "Only `create` writes a lite database ZIP. `priority` writes sidecars only. `analyse-priority` reads an existing CSV and does not rewrite the database."
        )
        safety_body.setObjectName("muted")
        safety_body.setWordWrap(True)
        safety_layout.addWidget(safety_title)
        safety_layout.addWidget(safety_body)
        root.addWidget(safety)

        for item in CLI_GUIDE_SECTIONS:
            card = QFrame()
            card.setObjectName("panel")
            layout = QVBoxLayout(card)
            layout.setContentsMargins(14, 12, 14, 12)
            title = QLabel(item["title"])
            title.setObjectName("sectionTitle")
            title.setStyleSheet("font-size: 16px;")
            summary = QLabel(item["summary"])
            summary.setObjectName("muted")
            summary.setWordWrap(True)
            command = QPlainTextEdit()
            command.setReadOnly(True)
            command.setPlainText(item["command"])
            command.setMaximumHeight(140)
            notes = QLabel("\n".join(f"- {note}" for note in item["notes"]))
            notes.setObjectName("muted")
            notes.setWordWrap(True)
            layout.addWidget(title)
            layout.addWidget(summary)
            layout.addWidget(command)
            layout.addWidget(notes)
            root.addWidget(card)
        root.addStretch(1)
        return tab

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

    def set_output_paths(self, lines: list[str]) -> None:
        self.output_box.setPlainText("\n".join(line for line in lines if line))

    def set_priority_output_paths(self, lines: list[str]) -> None:
        self.priority_output_box.setPlainText("\n".join(line for line in lines if line))

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

    def _set_run_controls_enabled(self, enabled: bool) -> None:
        if self.inspect_button is not None:
            self.inspect_button.setDisabled(not enabled)
        if self.create_button is not None:
            self.create_button.setDisabled(not enabled)
        if self.priority_button is not None:
            self.priority_button.setDisabled(not enabled)
        if self.use_database_methods_button is not None:
            self.use_database_methods_button.setDisabled((not enabled) or not self.database_has_impact_methods)
        if self.priority_use_database_methods_button is not None:
            self.priority_use_database_methods_button.setDisabled(
                (not enabled) or not self.priority_database_has_impact_methods
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

    def pick_cf_choices_load(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load CF choices",
            self.cf_choices_edit.text().strip(),
            "CSV files (*.csv)",
        )
        if path:
            self.cf_choices_edit.setText(path)
            self.append_status(f"Loaded CF choices file path: {path}")

    def pick_cf_choices_save(self) -> None:
        current_path = self.cf_choices_edit.text().strip()
        output_dir = self.output_edit.text().strip()
        default_path = current_path or str(Path(output_dir or ".") / "cf_resolution_choices.csv")
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save CF choices",
            default_path,
            "CSV files (*.csv)",
        )
        if not path:
            return
        destination = Path(path)
        if current_path:
            source = Path(current_path)
            if source.exists() and source.resolve() != destination.resolve():
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(source.read_bytes())
                self.append_status(f"Saved CF choices to {path}")
            elif source.exists():
                self.append_status(f"CF choices file already set to {path}")
            else:
                self.append_status(f"CF choices will be written to {path} on the next run.")
        else:
            self.append_status(f"CF choices will be written to {path} on the next run.")
        self.cf_choices_edit.setText(path)

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

    def pick_priority_cf_choices_load(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load CF choices",
            self.priority_cf_choices_edit.text().strip(),
            "CSV files (*.csv)",
        )
        if path:
            self.priority_cf_choices_edit.setText(path)
            self.append_priority_status(f"Loaded CF choices file path: {path}")

    def use_database_methods(self) -> None:
        self.methods_edit.clear()
        self.append_status("Using LCIA methods contained in the database. External methods input cleared.")

    def use_priority_database_methods(self) -> None:
        self.priority_methods_edit.clear()
        self.append_priority_status("Using LCIA methods contained in the database. External methods input cleared.")

    def prompt_cf_ambiguity(self, context: CFAmbiguityContext, candidates) -> CFPromptResult:
        dialog = CFAmbiguityDialog(
            self,
            context,
            [candidate_display_text(candidate) for candidate in candidates],
        )
        dialog.exec()
        return dialog.prompt_result

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

    @Slot(str)
    def _handle_worker_failure(self, message: str) -> None:
        QMessageBox.critical(self, "Run failed", message)
        if self._reduction_mode in {"priority", "priority_hint"}:
            self.append_priority_status(f"Run failed: {message}")
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
            cf_resolution_file=self.cf_choices_edit.text().strip() or None,
        )
        worker.progress.connect(self.update_create_progress)
        worker.prompt_requested.connect(self._handle_cf_prompt_request)
        worker.finished.connect(self._handle_create_result)
        worker.failed.connect(self._handle_worker_failure)
        self._start_reduction_worker(worker, mode="create")

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
            cf_resolution_file=self.priority_cf_choices_edit.text().strip() or None,
        )
        worker.progress.connect(self.update_priority_progress)
        worker.prompt_requested.connect(self._handle_cf_prompt_request)
        worker.finished.connect(self._handle_priority_result)
        worker.failed.connect(self._handle_worker_failure)
        self._start_reduction_worker(worker, mode="priority")

    @Slot(object, object)
    def _handle_cf_prompt_request(self, context: object, candidates: object) -> None:
        worker = self._reduction_worker
        if not isinstance(worker, (CreateWorker, PriorityWorker)):
            return
        prompt_result = self.prompt_cf_ambiguity(context, candidates)
        worker.deliver_prompt_result(prompt_result)

    @Slot(object)
    def _handle_create_result(self, result: object) -> None:
        output_text: list[str] = []
        if hasattr(result, "output_zip"):
            output_text.extend(
                [
                    result.output_zip,
                    result.exchange_manifest_csv,
                    result.process_manifest_csv,
                    result.run_summary_json,
                ]
            )
            self.set_output_paths(output_text)
            self.append_status(json.dumps(result.summary, indent=2, ensure_ascii=True))

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
