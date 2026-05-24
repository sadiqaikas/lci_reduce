"""PySide6 panel for analysing compact LCIA flow-priority CSV files."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .priority_analyser import (
    GroupBoundResult,
    PriorityAnalysisError,
    PriorityDataset,
    PriorityRecord,
    SelectionMatchResult,
    TauColumnPair,
    analyse_selected_group,
    build_priority_overview,
    build_priority_summary,
    load_priority_dataset,
    match_mixed_flow_items,
    parse_multiline_items,
    rank_rows_by_eta,
    rank_rows_by_loss_max,
    write_ranked_csv,
    write_selected_rows_csv,
    write_summary_json,
)


_EPSILON = 1e-12

TOP_HEADERS = ["", "flow_name", "flow_id", "eta_tau", "loss_max_tau", "tau_entry_min", "cf_status"]
RISK_HEADERS = ["", "flow_name", "flow_id", "eta_0_95", "eta_0_99", "loss_max_tau", "cf_status"]
CORE_TAIL_HEADERS = ["", "flow_name", "flow_id", "tau_entry_min", "eta_tau", "loss_max_tau"]
SELECTED_HEADERS = ["flow_name", "flow_id", "eta_tau", "loss_max_tau"]


class CollapsibleSection(QFrame):
    def __init__(self, title: str, content: QWidget, *, expanded: bool = False) -> None:
        super().__init__()
        self.setObjectName("panel")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(8)

        self.toggle = QToolButton()
        self.toggle.setText(title)
        self.toggle.setCheckable(True)
        self.toggle.setChecked(expanded)
        self.toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toggle.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self.toggle.setStyleSheet(
            "QToolButton { border: 0; font-weight: 700; font-size: 13px; text-align: left; padding: 0; }"
        )
        self.toggle.clicked.connect(self._toggle)

        self.content = content
        self.content.setVisible(expanded)
        layout.addWidget(self.toggle)
        layout.addWidget(self.content)

    def _toggle(self) -> None:
        expanded = self.toggle.isChecked()
        self.toggle.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self.content.setVisible(expanded)


class DistributionPlotCanvas(FigureCanvas):
    _FIGURE_WIDTH = 4.4
    _FIGURE_HEIGHT = 2.35
    _CANVAS_HEIGHT = 240

    def __init__(self, parent: QWidget | None = None) -> None:
        self.figure = Figure(figsize=(self._FIGURE_WIDTH, self._FIGURE_HEIGHT))
        self.axes = self.figure.add_subplot(111)
        super().__init__(self.figure)
        if parent is not None:
            self.setParent(parent)
        self.setMinimumHeight(self._CANVAS_HEIGHT)
        self.setMaximumHeight(self._CANVAS_HEIGHT)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.show_message("No data")

    def clear_figure(self) -> None:
        self.axes.clear()
        self.axes.set_axis_on()

    def _apply_compact_style(self) -> None:
        self.axes.tick_params(axis="both", labelsize=8)
        self.axes.xaxis.label.set_size(8)
        self.axes.yaxis.label.set_size(8)
        self.axes.title.set_size(9)
        self.figure.tight_layout(pad=0.8)

    def draw_plot(self, plotter: Callable) -> None:
        self.clear_figure()
        plotter(self.axes)
        self._apply_compact_style()
        self.draw()

    def show_message(self, message: str, *, title: str | None = None) -> None:
        self.clear_figure()
        if title:
            self.axes.set_title(title)
        self.axes.set_axis_off()
        self.axes.text(
            0.5,
            0.5,
            message,
            ha="center",
            va="center",
            fontsize=9,
            transform=self.axes.transAxes,
        )
        self.figure.tight_layout(pad=0.8)
        self.draw()


class PriorityAnalyserPanel(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.dataset: PriorityDataset | None = None
        self.selected_flow_ids: set[str] = set()
        self.last_group_result: GroupBoundResult | None = None
        self.last_selection_match: SelectionMatchResult | None = None

        self.top_rows: list[PriorityRecord] = []
        self.risk_rows: list[PriorityRecord] = []
        self.core_rows: list[PriorityRecord] = []
        self.tail_rows: list[PriorityRecord] = []
        self.selected_rows: list[PriorityRecord] = []

        self._updating_tables = False

        self.priority_csv_edit = QLineEdit()
        self.metadata_json_edit = QLineEdit()
        self.load_status_label = QLabel("Load a priority CSV to start.")
        self.load_status_label.setObjectName("muted")
        self.load_status_label.setWordWrap(True)
        self.selected_count_label = QLabel("Selected flows: 0")
        self.selected_count_label.setObjectName("muted")
        self.selected_count_label.setWordWrap(True)

        self.audit_tau_combo = QComboBox()
        self.tabs = QTabWidget()

        self.top_metric_combo = QComboBox()
        self.top_metric_combo.addItem("eta (selected tau)", "eta")
        self.top_metric_combo.addItem("loss_max (selected tau)", "loss")
        self.top_metric_combo.addItem("tau_entry_min", "tau_entry_min")
        self.top_metric_combo.addItem("tau_entry_median", "tau_entry_median")
        self.top_metric_combo.addItem("occurrence_count", "occurrence_count")
        self.top_search_edit = QLineEdit()
        self.top_search_edit.setPlaceholderText("Find flows by name or ID")
        self.top_limit_spin = QSpinBox()
        self.top_limit_spin.setRange(50, 5000)
        self.top_limit_spin.setSingleStep(50)
        self.top_limit_spin.setValue(200)
        self.top_cf_status_combo = QComboBox()
        self.top_cf_status_combo.addItem("all cf_status", "all")
        self.top_cf_status_combo.addItem("characterised", "characterised")
        self.top_cf_status_combo.addItem("partly_characterised", "partly_characterised")
        self.top_cf_status_combo.addItem("uncharacterised", "uncharacterised")
        self.top_eta_positive_checkbox = QCheckBox("eta > 0")
        self.top_loss_positive_checkbox = QCheckBox("loss_max > 0")
        self.top_status_label = QLabel("No priority file loaded.")
        self.top_status_label.setObjectName("muted")
        self.top_status_label.setWordWrap(True)
        self.show_consequence_button = QPushButton("Show consequence")
        self.show_consequence_button.clicked.connect(self.show_selected_consequence_dialog)
        self.top_table = QTableWidget(0, len(TOP_HEADERS))
        self.top_table.setHorizontalHeaderLabels(TOP_HEADERS)
        self.top_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.top_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.top_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.top_table.verticalHeader().setVisible(False)
        self.top_table.setAlternatingRowColors(False)
        self.top_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.top_detail_box = QPlainTextEdit()
        self.top_detail_box.setReadOnly(True)
        self.top_detail_box.setMaximumBlockCount(256)

        self.selection_summary_label = QLabel("Selected: 0")
        self.selection_summary_label.setWordWrap(True)
        self.selection_interval_label = QLabel("No flows selected.")
        self.selection_interval_label.setObjectName("muted")
        self.selection_interval_label.setWordWrap(True)
        self.selected_table = QTableWidget(0, len(SELECTED_HEADERS))
        self.selected_table.setHorizontalHeaderLabels(SELECTED_HEADERS)
        self.selected_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.selected_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.selected_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.selected_table.verticalHeader().setVisible(False)
        self.selected_table.setAlternatingRowColors(False)
        self.selection_result_box = QPlainTextEdit()
        self.selection_result_box.setReadOnly(True)
        self.selection_result_box.setMaximumBlockCount(512)
        self.selection_paste_box = QPlainTextEdit()
        self.selection_paste_box.setPlaceholderText("Optional: paste flow IDs or names, one per line")
        self.selection_paste_feedback_box = QPlainTextEdit()
        self.selection_paste_feedback_box.setReadOnly(True)
        self.selection_paste_feedback_box.setMaximumBlockCount(256)

        self.risk_class_combo = QComboBox()
        self.risk_class_combo.addItem("Critical at 0.95", "critical_095")
        self.risk_class_combo.addItem("Critical only at 0.99", "critical_only_099")
        self.risk_class_combo.addItem("Group-risk-only", "group_risk_only")
        self.risk_class_combo.addItem("Not visible to selected LCIA methods", "uncharacterised")
        self.risk_limit_spin = QSpinBox()
        self.risk_limit_spin.setRange(50, 5000)
        self.risk_limit_spin.setSingleStep(50)
        self.risk_limit_spin.setValue(200)
        self.risk_definition_label = QLabel("-")
        self.risk_definition_label.setObjectName("muted")
        self.risk_definition_label.setWordWrap(True)
        self.risk_status_label = QLabel("No priority file loaded.")
        self.risk_status_label.setObjectName("muted")
        self.risk_status_label.setWordWrap(True)
        self.risk_table = QTableWidget(0, len(RISK_HEADERS))
        self.risk_table.setHorizontalHeaderLabels(RISK_HEADERS)
        self.risk_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.risk_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.risk_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.risk_table.verticalHeader().setVisible(False)
        self.risk_table.setAlternatingRowColors(False)

        self.core_tail_limit_spin = QSpinBox()
        self.core_tail_limit_spin.setRange(20, 1000)
        self.core_tail_limit_spin.setSingleStep(20)
        self.core_tail_limit_spin.setValue(100)
        self.core_tail_status_label = QLabel("No priority file loaded.")
        self.core_tail_status_label.setObjectName("muted")
        self.core_tail_status_label.setWordWrap(True)
        self.core_table = QTableWidget(0, len(CORE_TAIL_HEADERS))
        self.core_table.setHorizontalHeaderLabels(CORE_TAIL_HEADERS)
        self.core_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.core_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.core_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.core_table.verticalHeader().setVisible(False)
        self.core_table.setAlternatingRowColors(False)
        self.tail_table = QTableWidget(0, len(CORE_TAIL_HEADERS))
        self.tail_table.setHorizontalHeaderLabels(CORE_TAIL_HEADERS)
        self.tail_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tail_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tail_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.tail_table.verticalHeader().setVisible(False)
        self.tail_table.setAlternatingRowColors(False)

        self.distributions_empty_label = QLabel("Load a priority CSV to view distributions.")
        self.distributions_empty_label.setObjectName("muted")
        self.distributions_empty_label.setAlignment(Qt.AlignCenter)
        self.distributions_empty_label.setWordWrap(True)
        self.distributions_empty_label.setMinimumHeight(160)
        self.distributions_empty_label.setMaximumHeight(160)
        self.distribution_histogram_bins = 10
        self.distribution_histogram_bins_min = 4
        self.distribution_histogram_bins_max = 40
        self.distribution_bins_label = QLabel()
        self.distribution_bins_label.setObjectName("muted")
        self.distribution_bins_label.setWordWrap(True)
        self.distribution_wider_bins_button = QPushButton("Wider bins")
        self.distribution_wider_bins_button.setObjectName("secondary")
        self.distribution_narrower_bins_button = QPushButton("Narrower bins")
        self.distribution_narrower_bins_button.setObjectName("secondary")
        self.distribution_eta_summary_label = QLabel("No priority file loaded.")
        self.distribution_eta_summary_label.setObjectName("muted")
        self.distribution_eta_summary_label.setWordWrap(True)
        self.distribution_loss_summary_label = QLabel("No priority file loaded.")
        self.distribution_loss_summary_label.setObjectName("muted")
        self.distribution_loss_summary_label.setWordWrap(True)
        self.distribution_eta_chart = self._create_distribution_plot_placeholder()
        self.distribution_loss_chart = self._create_distribution_plot_placeholder()
        self.distribution_chart_panels: list[DistributionPlotCanvas] = [
            self.distribution_eta_chart,
            self.distribution_loss_chart,
        ]

        self._build()
        self._wire_events()
        self._configure_tables()
        self._update_risk_definition()
        self._update_distribution_bin_controls()
        self._set_empty_state()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(12)

        root.addWidget(self._build_header_bar())
        root.addWidget(self._build_file_bar())

        self.tabs.addTab(self._build_priorities_tab(), "Priority flows")
        self.tabs.addTab(self._build_risk_classes_tab(), "Risk classes")
        self.tabs.addTab(self._build_core_tail_tab(), "Core vs tail")
        self.tabs.addTab(self._build_distributions_tab(), "Distributions")
        root.addWidget(self.tabs, 1)

    def _build_header_bar(self) -> QWidget:
        frame = QFrame()
        frame.setObjectName("panel")
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(14, 10, 14, 10)
        title = QLabel("Priority analyser")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)
        layout.addStretch(1)
        layout.addWidget(QLabel("Audit tau"))
        layout.addWidget(self.audit_tau_combo)
        return frame

    def _build_file_bar(self) -> QWidget:
        frame = QFrame()
        frame.setObjectName("panel")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)

        row = QGridLayout()
        row.setHorizontalSpacing(8)
        row.setVerticalSpacing(8)
        row.addWidget(QLabel("Priority CSV"), 0, 0)
        row.addWidget(self.priority_csv_edit, 0, 1)
        browse_csv_button = QPushButton("Browse")
        browse_csv_button.setObjectName("secondary")
        browse_csv_button.clicked.connect(self._pick_priority_csv)
        row.addWidget(browse_csv_button, 0, 2)
        row.addWidget(QLabel("Metadata JSON"), 0, 3)
        row.addWidget(self.metadata_json_edit, 0, 4)
        browse_json_button = QPushButton("Browse")
        browse_json_button.setObjectName("secondary")
        browse_json_button.clicked.connect(self._pick_metadata_json)
        row.addWidget(browse_json_button, 0, 5)
        load_button = QPushButton("Load")
        load_button.clicked.connect(self.load_priority_file)
        row.addWidget(load_button, 0, 6)
        clear_button = QPushButton("Clear")
        clear_button.setObjectName("secondary")
        clear_button.clicked.connect(self.clear_loaded_file)
        row.addWidget(clear_button, 0, 7)
        layout.addLayout(row)
        layout.addWidget(self.load_status_label)
        layout.addWidget(self.selected_count_label)
        return frame

    def _build_priorities_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        controls = QFrame()
        controls.setObjectName("panel")
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(12, 10, 12, 10)
        controls_layout.setSpacing(8)

        control_row = QGridLayout()
        control_row.setHorizontalSpacing(8)
        control_row.setVerticalSpacing(8)
        control_row.addWidget(QLabel("Rank by"), 0, 0)
        control_row.addWidget(self.top_metric_combo, 0, 1)
        control_row.addWidget(QLabel("Rows"), 0, 2)
        control_row.addWidget(self.top_limit_spin, 0, 3)
        control_row.addWidget(QLabel("Find"), 0, 4)
        control_row.addWidget(self.top_search_edit, 0, 5)
        control_row.addWidget(self.show_consequence_button, 0, 6)
        controls_layout.addLayout(control_row)
        controls_layout.addWidget(
            CollapsibleSection("Additional filters", self._build_top_extra_filters(), expanded=False)
        )
        controls_layout.addWidget(self.top_status_label)
        layout.addWidget(controls)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        table_frame = QFrame()
        table_frame.setObjectName("panel")
        table_layout = QVBoxLayout(table_frame)
        table_layout.setContentsMargins(10, 10, 10, 10)
        table_layout.setSpacing(8)
        table_layout.addWidget(self.top_table, 1)
        top_buttons = QHBoxLayout()
        export_button = QPushButton("Export top table CSV")
        export_button.setObjectName("secondary")
        export_button.clicked.connect(self.export_top_table_csv)
        top_buttons.addStretch(1)
        top_buttons.addWidget(export_button)
        table_layout.addLayout(top_buttons)
        splitter.addWidget(table_frame)

        side_panel = QWidget()
        side_layout = QVBoxLayout(side_panel)
        side_layout.setContentsMargins(0, 0, 0, 0)
        side_layout.setSpacing(10)

        detail_frame = QFrame()
        detail_frame.setObjectName("panel")
        detail_layout = QVBoxLayout(detail_frame)
        detail_layout.setContentsMargins(10, 10, 10, 10)
        detail_title = QLabel("Flow detail")
        detail_title.setObjectName("groupTitle")
        detail_layout.addWidget(detail_title)
        detail_layout.addWidget(self.top_detail_box, 1)
        side_layout.addWidget(detail_frame, 1)

        side_layout.addWidget(self._build_selection_panel(), 2)
        splitter.addWidget(side_panel)
        splitter.setSizes([920, 520])
        layout.addWidget(splitter, 1)
        return tab

    def _build_top_extra_filters(self) -> QWidget:
        widget = QWidget()
        layout = QGridLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(8)
        layout.addWidget(QLabel("cf_status"), 0, 0)
        layout.addWidget(self.top_cf_status_combo, 0, 1)
        layout.addWidget(self.top_eta_positive_checkbox, 0, 2)
        layout.addWidget(self.top_loss_positive_checkbox, 0, 3)
        return widget

    def _build_selection_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("panel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        title = QLabel("Selected flow effect")
        title.setObjectName("groupTitle")
        layout.addWidget(title)
        layout.addWidget(self.selection_summary_label)
        layout.addWidget(self.selection_interval_label)

        selected_frame = QFrame()
        selected_frame.setObjectName("groupCard")
        selected_layout = QVBoxLayout(selected_frame)
        selected_layout.setContentsMargins(10, 10, 10, 10)
        selected_layout.setSpacing(8)
        selected_buttons = QHBoxLayout()
        remove_button = QPushButton("Remove highlighted")
        remove_button.setObjectName("secondary")
        remove_button.clicked.connect(self.remove_highlighted_selected_rows)
        clear_button = QPushButton("Clear selection")
        clear_button.setObjectName("secondary")
        clear_button.clicked.connect(self.clear_selected_flows)
        export_csv_button = QPushButton("Export selection CSV")
        export_csv_button.setObjectName("secondary")
        export_csv_button.clicked.connect(self.export_selected_flow_csv)
        export_json_button = QPushButton("Export analysis JSON")
        export_json_button.setObjectName("secondary")
        export_json_button.clicked.connect(self.export_selected_analysis_json)
        selected_buttons.addWidget(remove_button)
        selected_buttons.addWidget(clear_button)
        selected_buttons.addStretch(1)
        selected_buttons.addWidget(export_csv_button)
        selected_buttons.addWidget(export_json_button)
        selected_layout.addLayout(selected_buttons)
        selected_layout.addWidget(self.selected_table)
        layout.addWidget(selected_frame, 1)

        result_frame = QFrame()
        result_frame.setObjectName("groupCard")
        result_layout = QVBoxLayout(result_frame)
        result_layout.setContentsMargins(10, 10, 10, 10)
        result_title = QLabel("Consequence")
        result_title.setObjectName("groupTitle")
        result_layout.addWidget(result_title)
        result_layout.addWidget(self.selection_result_box, 1)
        layout.addWidget(result_frame, 1)

        paste_widget = QWidget()
        paste_layout = QVBoxLayout(paste_widget)
        paste_layout.setContentsMargins(0, 0, 0, 0)
        paste_layout.setSpacing(8)
        paste_layout.addWidget(self.selection_paste_box)
        paste_buttons = QHBoxLayout()
        add_pasted_button = QPushButton("Add pasted items")
        add_pasted_button.clicked.connect(self.add_pasted_selection)
        clear_paste_button = QPushButton("Clear paste box")
        clear_paste_button.setObjectName("secondary")
        clear_paste_button.clicked.connect(self.selection_paste_box.clear)
        paste_buttons.addWidget(add_pasted_button)
        paste_buttons.addWidget(clear_paste_button)
        paste_buttons.addStretch(1)
        paste_layout.addLayout(paste_buttons)
        paste_layout.addWidget(self.selection_paste_feedback_box)
        layout.addWidget(CollapsibleSection("Paste flow IDs or names", paste_widget, expanded=False))
        return panel

    def _build_risk_classes_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        controls = QFrame()
        controls.setObjectName("panel")
        controls_layout = QGridLayout(controls)
        controls_layout.setContentsMargins(12, 10, 12, 10)
        controls_layout.setHorizontalSpacing(8)
        controls_layout.setVerticalSpacing(8)
        controls_layout.addWidget(QLabel("Risk class"), 0, 0)
        controls_layout.addWidget(self.risk_class_combo, 0, 1)
        controls_layout.addWidget(QLabel("Rows"), 0, 2)
        controls_layout.addWidget(self.risk_limit_spin, 0, 3)
        controls_layout.addWidget(self.risk_definition_label, 1, 0, 1, 4)
        controls_layout.addWidget(self.risk_status_label, 2, 0, 1, 4)
        layout.addWidget(controls)

        risk_frame = QFrame()
        risk_frame.setObjectName("panel")
        risk_layout = QVBoxLayout(risk_frame)
        risk_layout.setContentsMargins(10, 10, 10, 10)
        risk_layout.setSpacing(8)
        risk_layout.addWidget(self.risk_table, 1)
        layout.addWidget(risk_frame, 1)
        return tab

    def _build_core_tail_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        controls = QFrame()
        controls.setObjectName("panel")
        controls_layout = QGridLayout(controls)
        controls_layout.setContentsMargins(12, 10, 12, 10)
        controls_layout.setHorizontalSpacing(8)
        controls_layout.setVerticalSpacing(8)
        controls_layout.addWidget(QLabel("Rows per table"), 0, 0)
        controls_layout.addWidget(self.core_tail_limit_spin, 0, 1)
        definition = QLabel(
            "Core flows are the earliest non-null `tau_entry_min` flows. Tail flows are the latest non-null `tau_entry_min` flows."
        )
        definition.setObjectName("muted")
        definition.setWordWrap(True)
        controls_layout.addWidget(definition, 1, 0, 1, 4)
        controls_layout.addWidget(self.core_tail_status_label, 2, 0, 1, 4)
        layout.addWidget(controls)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        core_frame = QFrame()
        core_frame.setObjectName("panel")
        core_layout = QVBoxLayout(core_frame)
        core_layout.setContentsMargins(10, 10, 10, 10)
        core_layout.setSpacing(8)
        core_title = QLabel("Core flows")
        core_title.setObjectName("groupTitle")
        core_layout.addWidget(core_title)
        core_layout.addWidget(self.core_table, 1)
        splitter.addWidget(core_frame)

        tail_frame = QFrame()
        tail_frame.setObjectName("panel")
        tail_layout = QVBoxLayout(tail_frame)
        tail_layout.setContentsMargins(10, 10, 10, 10)
        tail_layout.setSpacing(8)
        tail_title = QLabel("Tail flows")
        tail_title.setObjectName("groupTitle")
        tail_layout.addWidget(tail_title)
        tail_layout.addWidget(self.tail_table, 1)
        splitter.addWidget(tail_frame)
        splitter.setSizes([780, 780])
        layout.addWidget(splitter, 1)
        return tab

    def _build_distributions_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        intro = QFrame()
        intro.setObjectName("panel")
        intro_layout = QVBoxLayout(intro)
        intro_layout.setContentsMargins(12, 10, 12, 10)
        intro_layout.setSpacing(8)
        message = QLabel(
            "Read-only eta and loss_max histograms from the loaded priority CSV for the currently selected audit tau."
        )
        message.setObjectName("muted")
        message.setWordWrap(True)
        intro_layout.addWidget(message)
        controls = QHBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(8)
        controls.addWidget(QLabel("Bin size"))
        controls.addWidget(self.distribution_wider_bins_button)
        controls.addWidget(self.distribution_narrower_bins_button)
        controls.addWidget(self.distribution_bins_label, 1)
        intro_layout.addLayout(controls)
        layout.addWidget(intro)

        self.distributions_grid = QWidget()
        self.distributions_grid.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        grid = QGridLayout(self.distributions_grid)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        grid.addWidget(
            self._build_distribution_card("eta histogram", self.distribution_eta_summary_label, self.distribution_eta_chart),
            0,
            0,
        )
        grid.addWidget(
            self._build_distribution_card(
                "loss_max histogram",
                self.distribution_loss_summary_label,
                self.distribution_loss_chart,
            ),
            0,
            1,
        )
        layout.addWidget(self.distributions_empty_label)
        layout.addWidget(self.distributions_grid)
        layout.addStretch(1)
        return tab

    def _build_distribution_card(self, title: str, summary_label: QLabel, chart_widget: QWidget) -> QWidget:
        frame = QFrame()
        frame.setObjectName("panel")
        frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        title_label = QLabel(title)
        title_label.setObjectName("groupTitle")
        layout.addWidget(title_label)
        summary_label.setMaximumHeight(36)
        layout.addWidget(summary_label)
        layout.addWidget(chart_widget, 0)
        return frame

    def _create_distribution_plot_placeholder(self) -> DistributionPlotCanvas:
        return DistributionPlotCanvas()

    def _wire_events(self) -> None:
        self.audit_tau_combo.currentIndexChanged.connect(self._refresh_all_views)
        self.top_metric_combo.currentIndexChanged.connect(self.refresh_top_flows)
        self.top_search_edit.textChanged.connect(self.refresh_top_flows)
        self.top_limit_spin.valueChanged.connect(self.refresh_top_flows)
        self.top_cf_status_combo.currentIndexChanged.connect(self.refresh_top_flows)
        self.top_eta_positive_checkbox.toggled.connect(self.refresh_top_flows)
        self.top_loss_positive_checkbox.toggled.connect(self.refresh_top_flows)
        self.top_table.itemSelectionChanged.connect(self.refresh_top_detail)
        self.top_table.itemChanged.connect(self._handle_checkbox_change)

        self.selected_table.itemSelectionChanged.connect(self._noop)

        self.risk_class_combo.currentIndexChanged.connect(self.refresh_risk_class_table)
        self.risk_limit_spin.valueChanged.connect(self.refresh_risk_class_table)
        self.risk_table.itemChanged.connect(self._handle_checkbox_change)

        self.core_tail_limit_spin.valueChanged.connect(self.refresh_core_tail_tables)
        self.core_table.itemChanged.connect(self._handle_checkbox_change)
        self.tail_table.itemChanged.connect(self._handle_checkbox_change)
        self.distribution_wider_bins_button.clicked.connect(self._make_distribution_bins_wider)
        self.distribution_narrower_bins_button.clicked.connect(self._make_distribution_bins_narrower)

    def _configure_tables(self) -> None:
        for table, stretch_column in [
            (self.top_table, 1),
            (self.risk_table, 1),
            (self.core_table, 1),
            (self.tail_table, 1),
            (self.selected_table, 0),
        ]:
            header = table.horizontalHeader()
            for column in range(table.columnCount()):
                mode = QHeaderView.ResizeToContents
                if column == stretch_column:
                    mode = QHeaderView.Stretch
                header.setSectionResizeMode(column, mode)
        for table in (self.core_table, self.tail_table):
            header = table.horizontalHeader()
            header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
            header.setSectionResizeMode(1, QHeaderView.Interactive)
            header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
            header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
            header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
            header.setSectionResizeMode(5, QHeaderView.ResizeToContents)
            table.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
            table.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            table.setWordWrap(False)
            table.setColumnWidth(1, 360)

    def _noop(self) -> None:
        return None

    def _set_empty_state(self) -> None:
        self.top_table.setRowCount(0)
        self.risk_table.setRowCount(0)
        self.core_table.setRowCount(0)
        self.tail_table.setRowCount(0)
        self.selected_table.setRowCount(0)
        self.top_detail_box.setPlainText("Load a priority CSV, then click a flow.")
        self.selection_result_box.setPlainText("Select flows to analyse a basket.")
        self.selection_summary_label.setText("Selected: 0")
        self.selection_interval_label.setText("No flows selected.")
        self.top_status_label.setText("No priority file loaded.")
        self.risk_status_label.setText("No priority file loaded.")
        self.core_tail_status_label.setText("No priority file loaded.")
        self.selected_count_label.setText("Selected flows: 0")
        self._clear_table_selection_visuals(self.top_table)
        self._clear_table_selection_visuals(self.risk_table)
        self._clear_table_selection_visuals(self.core_table)
        self._clear_table_selection_visuals(self.tail_table)
        self.distributions_empty_label.setVisible(True)
        self.distributions_grid.setVisible(False)
        self.distribution_eta_summary_label.setText("No priority file loaded.")
        self.distribution_loss_summary_label.setText("No priority file loaded.")
        self._clear_distribution_charts()
        self._update_distribution_bin_controls()

    def _update_distribution_bin_controls(self) -> None:
        self.distribution_bins_label.setText(f"{self.distribution_histogram_bins} bins shared across both histograms.")
        has_loaded_dataset = self.dataset is not None and self.current_tau_pair() is not None
        self.distribution_wider_bins_button.setEnabled(
            has_loaded_dataset and self.distribution_histogram_bins > self.distribution_histogram_bins_min
        )
        self.distribution_narrower_bins_button.setEnabled(
            has_loaded_dataset and self.distribution_histogram_bins < self.distribution_histogram_bins_max
        )

    def _set_distribution_histogram_bins(self, bins: int) -> None:
        clamped = max(self.distribution_histogram_bins_min, min(self.distribution_histogram_bins_max, int(bins)))
        if clamped == self.distribution_histogram_bins:
            self._update_distribution_bin_controls()
            return
        self.distribution_histogram_bins = clamped
        self._update_distribution_bin_controls()
        self.refresh_distributions()

    def _make_distribution_bins_wider(self) -> None:
        self._set_distribution_histogram_bins(self.distribution_histogram_bins - 1)

    def _make_distribution_bins_narrower(self) -> None:
        self._set_distribution_histogram_bins(self.distribution_histogram_bins + 1)

    def _pick_priority_csv(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select priority CSV",
            self.priority_csv_edit.text().strip(),
            "CSV files (*.csv)",
        )
        if path:
            self.priority_csv_edit.setText(path)

    def _pick_metadata_json(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select metadata JSON",
            self.metadata_json_edit.text().strip(),
            "JSON files (*.json)",
        )
        if path:
            self.metadata_json_edit.setText(path)

    def load_priority_file(self) -> None:
        priority_csv = self.priority_csv_edit.text().strip()
        metadata_json = self.metadata_json_edit.text().strip() or None
        if not priority_csv:
            QMessageBox.warning(self, "Missing input", "Select `lcia_flow_priority.csv` first.")
            return
        try:
            dataset = load_priority_dataset(priority_csv, metadata_json)
        except PriorityAnalysisError as exc:
            self.load_status_label.setText(f"Load failed: {exc}")
            QMessageBox.critical(self, "Priority file load failed", str(exc))
            return
        self.dataset = dataset
        self.selected_flow_ids.clear()
        self.last_group_result = None
        self.last_selection_match = None
        self._load_tau_options(dataset)
        self._update_header_status(dataset)
        self._update_risk_definition()
        self._refresh_all_views()

    def clear_loaded_file(self) -> None:
        self.dataset = None
        self.selected_flow_ids.clear()
        self.last_group_result = None
        self.last_selection_match = None
        self.audit_tau_combo.clear()
        self.load_status_label.setText("Load a priority CSV to start.")
        self.selection_paste_box.clear()
        self.selection_paste_feedback_box.clear()
        self._set_empty_state()

    def _load_tau_options(self, dataset: PriorityDataset) -> None:
        self.audit_tau_combo.blockSignals(True)
        self.audit_tau_combo.clear()
        preferred = dataset.get_tau_pair(None).token
        for pair in dataset.tau_pairs:
            label = f"{pair.tau_label} ({pair.eta_column} / {pair.loss_max_column})"
            self.audit_tau_combo.addItem(label, pair.token)
        index = self.audit_tau_combo.findData(preferred)
        if index >= 0:
            self.audit_tau_combo.setCurrentIndex(index)
        self.audit_tau_combo.blockSignals(False)

    def _update_header_status(self, dataset: PriorityDataset) -> None:
        overview = build_priority_overview(dataset)
        tau_text = ", ".join(pair.tau_label for pair in dataset.tau_pairs)
        self.load_status_label.setText(
            f"Loaded {Path(dataset.source_path).name} | tau: {tau_text} | flows: {overview['total_flows']:,}"
        )
        self.selected_count_label.setText("Selected flows: 0")

    def current_tau_pair(self) -> TauColumnPair | None:
        if self.dataset is None:
            return None
        token = self.audit_tau_combo.currentData()
        if token is None:
            return self.dataset.get_tau_pair(None)
        for pair in self.dataset.tau_pairs:
            if pair.token == token:
                return pair
        return self.dataset.get_tau_pair(None)

    def pair_095(self) -> TauColumnPair | None:
        if self.dataset is None or not self.dataset.has_tau(0.95):
            return None
        return self.dataset.get_tau_pair(0.95)

    def pair_099(self) -> TauColumnPair | None:
        if self.dataset is None or not self.dataset.has_tau(0.99):
            return None
        return self.dataset.get_tau_pair(0.99)

    def _refresh_all_views(self) -> None:
        self.refresh_top_flows()
        self.refresh_selected_basket()
        self.refresh_risk_class_table()
        self.refresh_core_tail_tables()
        self.refresh_distributions()

    def refresh_top_flows(self) -> None:
        if self.dataset is None:
            self.top_table.setRowCount(0)
            self.top_detail_box.setPlainText("Load a priority CSV, then click a flow.")
            return
        pair = self.current_tau_pair()
        if pair is None:
            return
        rows = list(self.dataset.rows)
        search = self.top_search_edit.text().strip().casefold()
        if search:
            rows = [
                row
                for row in rows
                if search in row.flow_name.casefold() or search in row.flow_id.casefold()
            ]
        cf_status = str(self.top_cf_status_combo.currentData())
        if cf_status != "all":
            rows = [row for row in rows if row.cf_status == cf_status]
        if self.top_eta_positive_checkbox.isChecked():
            rows = [row for row in rows if row.eta(pair) > _EPSILON]
        if self.top_loss_positive_checkbox.isChecked():
            rows = [row for row in rows if row.loss_max(pair) > _EPSILON]

        metric = self.top_metric_combo.currentData()
        if metric == "loss":
            rows = rank_rows_by_loss_max(rows, pair)
        elif metric == "tau_entry_min":
            rows = sorted(
                rows,
                key=lambda row: (
                    row.tau_entry_min is None,
                    float("inf") if row.tau_entry_min is None else row.tau_entry_min,
                    -row.eta(pair),
                    row.flow_id,
                    row.original_order,
                ),
            )
        elif metric == "tau_entry_median":
            rows = sorted(
                rows,
                key=lambda row: (
                    row.tau_entry_median is None,
                    float("inf") if row.tau_entry_median is None else row.tau_entry_median,
                    -row.eta(pair),
                    row.flow_id,
                    row.original_order,
                ),
            )
        elif metric == "occurrence_count":
            rows = sorted(
                rows,
                key=lambda row: (-row.occurrence_count, -row.eta(pair), row.flow_id, row.original_order),
            )
        else:
            rows = rank_rows_by_eta(rows, pair)

        self.top_rows = rows[: self.top_limit_spin.value()]
        self._render_top_table(pair)
        self.top_status_label.setText(
            f"Showing {len(self.top_rows):,} top flows from {len(rows):,} matching flows."
        )
        self.refresh_top_detail()

    def _render_top_table(self, pair: TauColumnPair) -> None:
        self._updating_tables = True
        self.top_table.setUpdatesEnabled(False)
        self.top_table.setRowCount(len(self.top_rows))
        try:
            for row_index, row in enumerate(self.top_rows):
                self._set_checkbox_item(self.top_table, row_index, row.flow_id)
                self._set_text_item(self.top_table, row_index, 1, row.flow_name)
                self._set_text_item(self.top_table, row_index, 2, row.flow_id)
                self._set_text_item(self.top_table, row_index, 3, _format_metric(row.eta(pair)), numeric=True)
                self._set_text_item(self.top_table, row_index, 4, _format_metric(row.loss_max(pair)), numeric=True)
                self._set_text_item(self.top_table, row_index, 5, _format_metric(row.tau_entry_min), numeric=True)
                self._set_text_item(self.top_table, row_index, 6, row.cf_status)
        finally:
            self.top_table.setUpdatesEnabled(True)
            self._updating_tables = False

    def refresh_top_detail(self) -> None:
        pair = self.current_tau_pair()
        if pair is None:
            self.top_detail_box.setPlainText("Load a priority CSV, then click a flow.")
            return
        selected_rows = self._selected_rows_from_table(self.top_rows, self.top_table)
        row = selected_rows[0] if selected_rows else self._current_row(self.top_rows, self.top_table)
        if row is None:
            self.top_detail_box.setPlainText("Load a priority CSV, then click a flow.")
            return
        self.top_detail_box.setPlainText(
            "\n".join(
                [
                    f"Flow name: {row.flow_name or '-'}",
                    f"Flow ID: {row.flow_id}",
                    f"Compartment: {row.compartment or '-'}",
                    f"Subcompartment: {row.subcompartment or '-'}",
                    f"Reference unit: {row.reference_unit or '-'}",
                    f"cf_status: {row.cf_status}",
                    f"Occurrences: {_format_int(row.occurrence_count)}",
                    f"Characterised occurrences: {_format_int(row.characterised_occurrence_count)}",
                    "",
                    f"eta ({pair.tau_label}): {_format_metric(row.eta(pair))}",
                    f"loss_max ({pair.tau_label}): {_format_metric(row.loss_max(pair))}",
                    f"tau_entry_min: {_format_metric(row.tau_entry_min)}",
                    f"tau_entry_median: {_format_metric(row.tau_entry_median)}",
                    f"tau_entry_max: {_format_metric(row.tau_entry_max)}",
                ]
            )
        )

    def refresh_selected_basket(self) -> None:
        pair = self.current_tau_pair()
        if self.dataset is None or pair is None:
            self.selected_table.setRowCount(0)
            self.selection_summary_label.setText("Selected: 0")
            self.selection_interval_label.setText("No flows selected.")
            self.selection_result_box.setPlainText("Select flows to analyse a basket.")
            return

        self.selected_rows = [
            self.dataset.rows_by_flow_id[flow_id]
            for flow_id in sorted(self.selected_flow_ids)
            if flow_id in self.dataset.rows_by_flow_id
        ]
        self.selected_rows = rank_rows_by_eta(self.selected_rows, pair)
        self._render_selected_table(pair)

        result = analyse_selected_group(self.selected_rows, pair)
        self.last_group_result = result
        summary_parts = [f"Selected: {result.selected_count:,}", f"Sum loss_max: {_format_metric(result.sum_loss_max)}"]
        if result.exact_eta is not None:
            summary_parts.append(f"Exact eta: {_format_metric(result.exact_eta)}")
        else:
            summary_parts.extend(
                [
                    f"Lower bound: {_format_metric(result.lower_bound)}",
                    f"Upper bound: {_format_metric(result.upper_bound)}",
                ]
            )
        self.selection_summary_label.setText(" | ".join(summary_parts))
        self.selection_interval_label.setText(result.interval_text)
        self.selection_result_box.setPlainText(self._format_group_result(result))
        self.selected_count_label.setText(f"Selected flows: {result.selected_count:,}")

    def _render_selected_table(self, pair: TauColumnPair) -> None:
        self.selected_table.setUpdatesEnabled(False)
        self.selected_table.setRowCount(len(self.selected_rows))
        try:
            for row_index, row in enumerate(self.selected_rows):
                self._set_text_item(self.selected_table, row_index, 0, row.flow_name)
                self._set_text_item(self.selected_table, row_index, 1, row.flow_id)
                self._set_text_item(self.selected_table, row_index, 2, _format_metric(row.eta(pair)), numeric=True)
                self._set_text_item(self.selected_table, row_index, 3, _format_metric(row.loss_max(pair)), numeric=True)
        finally:
            self.selected_table.setUpdatesEnabled(True)

    def refresh_risk_class_table(self) -> None:
        if self.dataset is None:
            self.risk_table.setRowCount(0)
            return
        current_pair = self.current_tau_pair()
        if current_pair is None:
            return
        class_key = self.risk_class_combo.currentData()
        rows, status = self._risk_class_rows(class_key, current_pair)
        self.risk_rows = rows[: self.risk_limit_spin.value()]
        self._render_risk_table(current_pair)
        self.risk_status_label.setText(status)
        self._update_risk_definition()

    def _risk_class_rows(self, class_key: str, current_pair: TauColumnPair) -> tuple[list[PriorityRecord], str]:
        pair_095 = self.pair_095()
        pair_099 = self.pair_099()
        if class_key == "critical_095":
            if pair_095 is None:
                return [], "0.95 columns are not available in this file."
            rows = [row for row in self.dataset.rows if row.eta(pair_095) > _EPSILON]
            return rank_rows_by_eta(rows, pair_095), f"{len(rows):,} flows are critical at 0.95."
        if class_key == "critical_only_099":
            if pair_095 is None or pair_099 is None:
                return [], "Both 0.95 and 0.99 columns are required for this view."
            rows = [row for row in self.dataset.rows if row.eta(pair_095) <= _EPSILON and row.eta(pair_099) > _EPSILON]
            return rank_rows_by_eta(rows, pair_099), f"{len(rows):,} flows are critical only at 0.99."
        if class_key == "group_risk_only":
            rows = [row for row in self.dataset.rows if row.eta(current_pair) <= _EPSILON and row.loss_max(current_pair) > _EPSILON]
            return rank_rows_by_loss_max(rows, current_pair), (
                f"{len(rows):,} flows are group-risk-only at audit tau {current_pair.tau_label}."
            )
        rows = [row for row in self.dataset.rows if row.cf_status == "uncharacterised"]
        return rows, f"{len(rows):,} flows are uncharacterised."

    def _render_risk_table(self, current_pair: TauColumnPair) -> None:
        pair_095 = self.pair_095()
        pair_099 = self.pair_099()
        self._updating_tables = True
        self.risk_table.setUpdatesEnabled(False)
        self.risk_table.setRowCount(len(self.risk_rows))
        try:
            for row_index, row in enumerate(self.risk_rows):
                self._set_checkbox_item(self.risk_table, row_index, row.flow_id)
                self._set_text_item(self.risk_table, row_index, 1, row.flow_name)
                self._set_text_item(self.risk_table, row_index, 2, row.flow_id)
                self._set_text_item(
                    self.risk_table,
                    row_index,
                    3,
                    _format_metric(row.eta(pair_095)) if pair_095 is not None else "-",
                    numeric=True,
                )
                self._set_text_item(
                    self.risk_table,
                    row_index,
                    4,
                    _format_metric(row.eta(pair_099)) if pair_099 is not None else "-",
                    numeric=True,
                )
                self._set_text_item(self.risk_table, row_index, 5, _format_metric(row.loss_max(current_pair)), numeric=True)
                self._set_text_item(self.risk_table, row_index, 6, row.cf_status)
        finally:
            self.risk_table.setUpdatesEnabled(True)
            self._updating_tables = False

    def _update_risk_definition(self) -> None:
        class_key = self.risk_class_combo.currentData()
        current_pair = self.current_tau_pair()
        if class_key == "critical_095":
            text = "Critical at 0.95: eta_0_95 > 0"
        elif class_key == "critical_only_099":
            text = "Critical only at 0.99: eta_0_95 = 0 and eta_0_99 > 0"
        elif class_key == "group_risk_only":
            tau_label = current_pair.tau_label if current_pair is not None else "selected tau"
            text = f"Group-risk-only: eta_{tau_label} = 0 and loss_max_{tau_label} > 0"
        else:
            text = "Not visible to selected LCIA methods: cf_status = uncharacterised"
        self.risk_definition_label.setText(text)

    def refresh_core_tail_tables(self) -> None:
        if self.dataset is None:
            self.core_table.setRowCount(0)
            self.tail_table.setRowCount(0)
            return
        pair = self.current_tau_pair()
        if pair is None:
            return
        eligible = [row for row in self.dataset.rows if row.tau_entry_min is not None]
        core_sorted = sorted(
            eligible,
            key=lambda row: (
                row.tau_entry_min,
                row.tau_entry_median if row.tau_entry_median is not None else float("inf"),
                row.flow_id,
                row.original_order,
            ),
        )
        tail_sorted = sorted(
            eligible,
            key=lambda row: (
                row.tau_entry_min,
                row.tau_entry_median if row.tau_entry_median is not None else float("inf"),
                row.flow_id,
                row.original_order,
            ),
            reverse=True,
        )
        limit = self.core_tail_limit_spin.value()
        self.core_rows = core_sorted[:limit]
        self.tail_rows = tail_sorted[:limit]
        self._render_core_tail_table(self.core_table, self.core_rows, pair)
        self._render_core_tail_table(self.tail_table, self.tail_rows, pair)
        self.core_tail_status_label.setText(
            f"{len(eligible):,} flows have non-null tau_entry_min. Showing {len(self.core_rows):,} core and {len(self.tail_rows):,} tail rows."
        )

    def refresh_distributions(self) -> None:
        if self.dataset is None:
            self.distributions_empty_label.setVisible(True)
            self.distributions_grid.setVisible(False)
            self.distribution_eta_summary_label.setText("No priority file loaded.")
            self.distribution_loss_summary_label.setText("No priority file loaded.")
            self._update_distribution_bin_controls()
            self._clear_distribution_charts()
            return
        pair = self.current_tau_pair()
        if pair is None:
            self._update_distribution_bin_controls()
            self._clear_distribution_charts()
            return
        self.distributions_empty_label.setVisible(False)
        self.distributions_grid.setVisible(True)
        rows = list(self.dataset.rows)
        eta_values = [row.eta(pair) for row in rows]
        loss_values = [row.loss_max(pair) for row in rows]
        self._update_distribution_bin_controls()

        self.distribution_eta_summary_label.setText(
            self._distribution_summary_text(
                total=len(eta_values),
                zero_count=sum(value <= _EPSILON for value in eta_values),
                positive_count=sum(value > _EPSILON for value in eta_values),
                saturated_count=sum(abs(value - pair.tau) <= _EPSILON for value in eta_values),
                saturated_label=f"eta = {pair.tau_label}",
            )
        )
        self.distribution_loss_summary_label.setText(
            self._distribution_summary_text(
                total=len(loss_values),
                zero_count=sum(value <= _EPSILON for value in loss_values),
                positive_count=sum(value > _EPSILON for value in loss_values),
                saturated_count=sum(abs(value - 1.0) <= _EPSILON for value in loss_values),
                saturated_label="loss_max = 1",
            )
        )
        self._populate_histogram_chart(
            self.distribution_eta_chart,
            eta_values,
            title=f"eta distribution ({pair.tau_label})",
            upper_bound=max(pair.tau, max(eta_values, default=0.0)),
            bins=self.distribution_histogram_bins,
        )
        self._populate_histogram_chart(
            self.distribution_loss_chart,
            loss_values,
            title=f"loss_max distribution ({pair.tau_label})",
            upper_bound=max(1.0, max(loss_values, default=0.0)),
            bins=self.distribution_histogram_bins,
        )

    def _distribution_summary_text(
        self,
        *,
        total: int,
        zero_count: int,
        positive_count: int,
        saturated_count: int | None,
        saturated_label: str | None,
        non_null_count: int | None = None,
    ) -> str:
        parts = [f"Rows: {total:,}"]
        if non_null_count is not None:
            parts.append(f"non-null: {non_null_count:,}")
        parts.append(f"zero: {zero_count:,}")
        parts.append(f"positive: {positive_count:,}")
        if saturated_count is not None and saturated_label:
            parts.append(f"{saturated_label}: {saturated_count:,}")
        return " | ".join(parts)

    def _clear_distribution_charts(self) -> None:
        for chart in self.distribution_chart_panels:
            chart.show_message("No data")

    def _populate_histogram_chart(
        self,
        widget: DistributionPlotCanvas,
        values: list[float],
        *,
        title: str,
        upper_bound: float,
        bins: int,
    ) -> None:
        if not values:
            widget.show_message("No data", title=title)
            return
        bounded_upper = max(float(upper_bound), _EPSILON)

        def _plot(axes) -> None:
            axes.hist(values, bins=bins, range=(0.0, bounded_upper))
            axes.set_title(title)
            axes.set_xlim(0.0, bounded_upper)
            axes.set_ylabel("Count")
            axes.set_xlabel("Value")
            axes.set_ylim(bottom=0)

        widget.draw_plot(_plot)

    def _render_core_tail_table(
        self,
        table: QTableWidget,
        rows: list[PriorityRecord],
        pair: TauColumnPair,
    ) -> None:
        self._updating_tables = True
        table.setUpdatesEnabled(False)
        table.setRowCount(len(rows))
        try:
            for row_index, row in enumerate(rows):
                self._set_checkbox_item(table, row_index, row.flow_id)
                self._set_text_item(table, row_index, 1, row.flow_name)
                self._set_text_item(table, row_index, 2, row.flow_id)
                self._set_text_item(table, row_index, 3, _format_metric(row.tau_entry_min), numeric=True)
                self._set_text_item(table, row_index, 4, _format_metric(row.eta(pair)), numeric=True)
                self._set_text_item(table, row_index, 5, _format_metric(row.loss_max(pair)), numeric=True)
        finally:
            table.setUpdatesEnabled(True)
            self._updating_tables = False

    def _set_checkbox_item(self, table: QTableWidget, row_index: int, flow_id: str) -> None:
        item = QTableWidgetItem()
        item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
        item.setCheckState(Qt.Checked if flow_id in self.selected_flow_ids else Qt.Unchecked)
        item.setData(Qt.UserRole, flow_id)
        self._apply_selection_background(item, flow_id)
        table.setItem(row_index, 0, item)

    def _set_text_item(
        self,
        table: QTableWidget,
        row_index: int,
        column: int,
        text: str,
        *,
        numeric: bool = False,
    ) -> None:
        flow_id_item = table.item(row_index, 0)
        flow_id = flow_id_item.data(Qt.UserRole) if flow_id_item is not None else ""
        item = QTableWidgetItem(text)
        item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
        if numeric:
            item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._apply_selection_background(item, str(flow_id))
        table.setItem(row_index, column, item)

    def _apply_selection_background(self, item: QTableWidgetItem, flow_id: str) -> None:
        if flow_id in self.selected_flow_ids:
            item.setBackground(QColor("#d8efe5"))
        else:
            item.setBackground(QColor("#ffffff"))

    def _handle_checkbox_change(self, item: QTableWidgetItem) -> None:
        if self._updating_tables or item.column() != 0:
            return
        flow_id = item.data(Qt.UserRole)
        if not flow_id:
            return
        if item.checkState() == Qt.Checked:
            self.selected_flow_ids.add(str(flow_id))
        else:
            self.selected_flow_ids.discard(str(flow_id))
        self._refresh_selection_state()

    def _current_row(self, rows: list[PriorityRecord], table: QTableWidget) -> PriorityRecord | None:
        selection = table.selectionModel().selectedRows()
        if not selection:
            return rows[0] if rows else None
        row_index = selection[0].row()
        if 0 <= row_index < len(rows):
            return rows[row_index]
        return None

    def _selected_rows_from_table(
        self,
        rows: list[PriorityRecord],
        table: QTableWidget,
    ) -> list[PriorityRecord]:
        selected_rows: list[PriorityRecord] = []
        for model_index in table.selectionModel().selectedRows():
            row_index = model_index.row()
            if 0 <= row_index < len(rows):
                selected_rows.append(rows[row_index])
        return selected_rows

    def remove_highlighted_selected_rows(self) -> None:
        removed = 0
        for model_index in self.selected_table.selectionModel().selectedRows():
            row_index = model_index.row()
            if not (0 <= row_index < len(self.selected_rows)):
                continue
            flow_id = self.selected_rows[row_index].flow_id
            if flow_id in self.selected_flow_ids:
                self.selected_flow_ids.remove(flow_id)
                removed += 1
        if removed > 0:
            self._refresh_selection_state()

    def clear_selected_flows(self) -> None:
        self.selected_flow_ids.clear()
        self._refresh_selection_state()

    def add_pasted_selection(self) -> None:
        if self.dataset is None:
            QMessageBox.warning(self, "No priority file", "Load a priority CSV first.")
            return
        items = parse_multiline_items(self.selection_paste_box.toPlainText())
        if not items:
            QMessageBox.warning(self, "No pasted items", "Paste at least one flow ID or flow name.")
            return
        result = match_mixed_flow_items(self.dataset, items)
        self.last_selection_match = result
        for flow_id in result.matched_flow_ids:
            self.selected_flow_ids.add(flow_id)
        self.selection_paste_feedback_box.setPlainText(self._format_selection_feedback(result))
        self._refresh_selection_state()

    def export_top_table_csv(self) -> None:
        pair = self.current_tau_pair()
        if self.dataset is None or pair is None or not self.top_rows:
            QMessageBox.warning(self, "No rows", "There are no top-flow rows to export.")
            return
        default_path = Path(self.dataset.source_path).with_name("priority_top_flows.csv")
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export top flows CSV",
            str(default_path),
            "CSV files (*.csv)",
        )
        if not path:
            return
        write_ranked_csv(path, self.top_rows, pair)

    def export_selected_flow_csv(self) -> None:
        pair = self.current_tau_pair()
        if self.dataset is None or pair is None or not self.selected_rows:
            QMessageBox.warning(self, "No selection", "Select flows first.")
            return
        default_path = Path(self.dataset.source_path).with_name("priority_selected_flows.csv")
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export selected flows CSV",
            str(default_path),
            "CSV files (*.csv)",
        )
        if not path:
            return
        write_selected_rows_csv(path, self.selected_rows, pair)

    def export_selected_analysis_json(self) -> None:
        pair = self.current_tau_pair()
        if self.dataset is None or pair is None:
            QMessageBox.warning(self, "No priority file", "Load a priority file first.")
            return
        default_path = Path(self.dataset.source_path).with_name("priority_selected_analysis.json")
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export selected analysis JSON",
            str(default_path),
            "JSON files (*.json)",
        )
        if not path:
            return
        summary = build_priority_summary(
            self.dataset,
            pair,
            top_n=len(self.top_rows) or self.top_limit_spin.value(),
            filtered_rows=self.top_rows,
            selected_rows=self.selected_rows,
            selection_match=self.last_selection_match,
        )
        write_summary_json(path, summary)

    def show_selected_consequence_dialog(self) -> None:
        pair = self.current_tau_pair()
        if self.dataset is None or pair is None:
            QMessageBox.warning(self, "No priority file", "Load a priority CSV first.")
            return
        if not self.selected_rows:
            QMessageBox.warning(self, "No selected flows", "Select one or more flows first.")
            return
        result = analyse_selected_group(self.selected_rows, pair)
        dialog = QDialog(self)
        dialog.setWindowTitle("Selected flow consequence")
        dialog.resize(820, 560)
        layout = QVBoxLayout(dialog)
        summary = QLabel(
            f"Selected flows: {result.selected_count:,} | audit tau: {pair.tau_label}"
        )
        summary.setWordWrap(True)
        layout.addWidget(summary)
        text_box = QPlainTextEdit()
        text_box.setReadOnly(True)
        text_box.setPlainText(self._format_consequence_popup_text(result))
        layout.addWidget(text_box, 1)
        close_button = QPushButton("Close")
        close_button.clicked.connect(dialog.accept)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(close_button)
        layout.addLayout(buttons)
        dialog.exec()

    def _format_group_result(self, result: GroupBoundResult) -> str:
        lines = ["Selected-flow consequence", result.interval_text, ""]
        if result.exact_eta is not None:
            lines.append(f"Exact eta_F({result.pair.tau_label}) = {_format_metric(result.exact_eta)}")
            if result.exact_reason:
                lines.append(result.exact_reason)
        else:
            lines.extend(
                [
                    "Bound used from the compact priority CSV",
                    f"Lower bound = max eta_f({result.pair.tau_label}) = {_format_metric(result.lower_bound)}",
                    (
                        f"Upper bound = min({result.pair.tau_label}, sum loss_max_f({result.pair.tau_label})) = "
                        f"{_format_metric(result.upper_bound)}"
                    ),
                ]
            )
        lines.extend(["", "Interpretation"])
        for message in result.interpretation:
            lines.append(f"- {message}")
        lines.extend(["", "Top selected flows by eta"])
        for index, row in enumerate(result.ranked_by_eta[:10], start=1):
            lines.append(
                f"{index}. {row.flow_name} | {row.flow_id} | "
                f"eta={_format_metric(row.eta(result.pair))} | "
                f"loss_max={_format_metric(row.loss_max(result.pair))}"
            )
        return "\n".join(lines)

    def _format_selection_feedback(self, result: SelectionMatchResult) -> str:
        lines = [
            f"Matched: {len(result.matched_flow_ids)}",
            f"Unmatched: {len(result.unmatched_items)}",
            f"Ambiguous: {len(result.ambiguous_items)}",
        ]
        if result.unmatched_items:
            lines.extend(["", "Unmatched"])
            for item in result.unmatched_items:
                lines.append(f"- {item}")
        if result.ambiguous_items:
            lines.extend(["", "Ambiguous"])
            for item, candidates in result.ambiguous_items.items():
                lines.append(f"- {item}")
                for candidate in candidates:
                    lines.append(f"  {candidate}")
        return "\n".join(lines)

    def _format_consequence_popup_text(self, result: GroupBoundResult) -> str:
        tau_label = result.pair.tau_label
        lines = [
            "Selected-flow consequence",
            f"max eta_f({tau_label}) <= eta_F({tau_label}) <= min({tau_label}, sum loss_max_f({tau_label}))",
            "",
        ]
        if result.exact_eta is not None:
            lines.append(f"Exact eta_F({tau_label}) = {_format_metric(result.exact_eta)}")
            if result.exact_reason:
                lines.append(result.exact_reason)
        else:
            lines.extend(
                [
                    f"Lower bound = max eta_f({tau_label}) = {_format_metric(result.lower_bound)}",
                    f"Upper bound = min({tau_label}, sum loss_max_f({tau_label})) = {_format_metric(result.upper_bound)}",
                    f"Sum loss_max_f({tau_label}) = {_format_metric(result.sum_loss_max)}",
                    "",
                    "This selection is not exactly identifiable from the compact CSV, so the interval is shown.",
                ]
            )
        lines.extend(["", "Interpretation"])
        for message in result.interpretation:
            lines.append(f"- {message}")
        lines.extend(["", "Top selected flows by eta"])
        for index, row in enumerate(result.ranked_by_eta[:10], start=1):
            lines.append(
                f"{index}. {row.flow_name} | {row.flow_id} | "
                f"eta={_format_metric(row.eta(result.pair))} | "
                f"loss_max={_format_metric(row.loss_max(result.pair))}"
            )
        return "\n".join(lines)

    def _refresh_selection_state(self) -> None:
        self.refresh_selected_basket()
        self._sync_visible_selection_state()

    def _sync_visible_selection_state(self) -> None:
        self._updating_tables = True
        try:
            self._sync_table_selection_state(self.top_table)
            self._sync_table_selection_state(self.risk_table)
            self._sync_table_selection_state(self.core_table)
            self._sync_table_selection_state(self.tail_table)
        finally:
            self._updating_tables = False

    def _sync_table_selection_state(self, table: QTableWidget) -> None:
        table.setUpdatesEnabled(False)
        try:
            for row_index in range(table.rowCount()):
                checkbox_item = table.item(row_index, 0)
                if checkbox_item is None:
                    continue
                flow_id = str(checkbox_item.data(Qt.UserRole) or "")
                selected = flow_id in self.selected_flow_ids
                expected_state = Qt.Checked if selected else Qt.Unchecked
                if checkbox_item.checkState() != expected_state:
                    checkbox_item.setCheckState(expected_state)
                self._apply_selection_background(checkbox_item, flow_id)
                for column in range(1, table.columnCount()):
                    item = table.item(row_index, column)
                    if item is not None:
                        self._apply_selection_background(item, flow_id)
        finally:
            table.setUpdatesEnabled(True)

    def _clear_table_selection_visuals(self, table: QTableWidget) -> None:
        for row_index in range(table.rowCount()):
            for column in range(table.columnCount()):
                item = table.item(row_index, column)
                if item is not None:
                    item.setBackground(QColor("#ffffff"))


def _format_int(value: int | None) -> str:
    return "-" if value is None else f"{int(value):,}"


def _format_metric(value: float | None) -> str:
    if value is None:
        return "-"
    return format(float(value), ".6g")
