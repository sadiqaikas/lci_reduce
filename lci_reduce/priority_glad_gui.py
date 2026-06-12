"""PySide6 panel for direct priority-to-GLAD target matching."""

from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
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
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .manifest import write_manifest_csv
from .priority_analyser import PriorityAnalysisError, PriorityDataset, TauColumnPair
from .priority_glad import (
    PriorityGladRowResult,
    build_priority_glad_rows,
    priority_glad_output_columns,
    priority_glad_target_keys,
    serialise_priority_glad_row,
    write_priority_glad_outputs,
)


_TARGET_KEYS = priority_glad_target_keys()
_MATCHED_STATUSES = {"exact_consensus", "exact_uuid", "exact_name_context_unit"}
_UNRESOLVED_STATUSES = {"conflict", "ambiguous", "unmatched"}


class PriorityGladPanel(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.dataset: PriorityDataset | None = None
        self.selected_pair: TauColumnPair | None = None
        self.row_results: list[PriorityGladRowResult] = []
        self.filtered_results: list[PriorityGladRowResult] = []
        self.asset_manifest: dict[str, object] | None = None

        self.priority_csv_edit = QLineEdit()
        self.metadata_json_edit = QLineEdit()
        self.output_dir_edit = QLineEdit()
        self.audit_tau_combo = QComboBox()
        self.target_filter_combo = QComboBox()
        self.target_filter_combo.addItem("All targets", "all")
        for target_key in _TARGET_KEYS:
            self.target_filter_combo.addItem(target_key, target_key)
        self.status_filter_combo = QComboBox()
        self.status_filter_combo.addItem("All rows", "all")
        self.status_filter_combo.addItem("Matched only", "matched")
        self.status_filter_combo.addItem("Unmatched only", "unmatched")
        self.status_filter_combo.addItem("Ambiguous/conflict only", "problem")
        self.eta_positive_checkbox = QCheckBox("eta > 0")
        self.loss_positive_checkbox = QCheckBox("loss_max > 0")
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Find by flow name or UUID")
        self.status_label = QLabel("Load a priority CSV to start.")
        self.status_label.setObjectName("muted")
        self.status_label.setWordWrap(True)
        self.summary_label = QLabel("No priority file loaded.")
        self.summary_label.setObjectName("muted")
        self.summary_label.setWordWrap(True)
        self.table = QTableWidget(0, 11)
        self.table.setHorizontalHeaderLabels(
            [
                "flow_name",
                "flow_id",
                "eta_tau",
                "loss_max_tau",
                "ecoinventEFv3.7",
                "ILCD_EFv3.0",
                "FEDEFLv1.0.3",
                "IDEA_EFv2.3",
                "matched_target_uuid",
                "matched_target_name",
                "candidate_count",
            ]
        )
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(False)
        self.detail_box = QPlainTextEdit()
        self.detail_box.setReadOnly(True)
        self.detail_box.setMaximumBlockCount(512)

        self._build()
        self._wire_events()
        self._configure_table()
        self._set_empty_state()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(12)

        intro = QFrame()
        intro.setObjectName("panel")
        intro_layout = QVBoxLayout(intro)
        intro_layout.setContentsMargins(16, 14, 16, 14)
        title = QLabel("Priority -> GLAD target matching")
        title.setObjectName("sectionTitle")
        body = QLabel(
            "Read-only direct matching from an existing compact LCIA priority CSV to bundled GLAD target flow lists. "
            "Matching uses exact UUID or exact normalized Name+Context+Unit only."
        )
        body.setObjectName("muted")
        body.setWordWrap(True)
        intro_layout.addWidget(title)
        intro_layout.addWidget(body)
        root.addWidget(intro)

        inputs_group = QGroupBox("Inputs")
        inputs_form = QFormLayout(inputs_group)

        csv_row = QHBoxLayout()
        csv_row.addWidget(self.priority_csv_edit, 1)
        browse_csv_button = QPushButton("Browse")
        browse_csv_button.setObjectName("secondary")
        browse_csv_button.clicked.connect(self._pick_priority_csv)
        csv_row.addWidget(browse_csv_button)
        inputs_form.addRow("Priority CSV", self._wrap_layout(csv_row))

        metadata_row = QHBoxLayout()
        metadata_row.addWidget(self.metadata_json_edit, 1)
        browse_metadata_button = QPushButton("Browse")
        browse_metadata_button.setObjectName("secondary")
        browse_metadata_button.clicked.connect(self._pick_metadata_json)
        metadata_row.addWidget(browse_metadata_button)
        inputs_form.addRow("Metadata JSON", self._wrap_layout(metadata_row))

        output_row = QHBoxLayout()
        output_row.addWidget(self.output_dir_edit, 1)
        browse_output_button = QPushButton("Browse")
        browse_output_button.setObjectName("secondary")
        browse_output_button.clicked.connect(self._pick_output_dir)
        output_row.addWidget(browse_output_button)
        inputs_form.addRow("Output folder", self._wrap_layout(output_row))

        action_row = QHBoxLayout()
        load_button = QPushButton("Load and match")
        load_button.clicked.connect(self.load_matches)
        action_row.addWidget(load_button)
        export_button = QPushButton("Write all reports")
        export_button.setObjectName("secondary")
        export_button.clicked.connect(self.export_all_reports)
        action_row.addWidget(export_button)
        action_row.addStretch(1)
        inputs_form.addRow("", self._wrap_layout(action_row))
        root.addWidget(inputs_group)

        filters_group = QGroupBox("Filters")
        filters_layout = QGridLayout(filters_group)
        filters_layout.addWidget(QLabel("Audit tau"), 0, 0)
        filters_layout.addWidget(self.audit_tau_combo, 0, 1)
        filters_layout.addWidget(QLabel("Target"), 0, 2)
        filters_layout.addWidget(self.target_filter_combo, 0, 3)
        filters_layout.addWidget(QLabel("Status"), 0, 4)
        filters_layout.addWidget(self.status_filter_combo, 0, 5)
        filters_layout.addWidget(self.eta_positive_checkbox, 1, 0, 1, 2)
        filters_layout.addWidget(self.loss_positive_checkbox, 1, 2, 1, 2)
        filters_layout.addWidget(QLabel("Search"), 1, 4)
        filters_layout.addWidget(self.search_edit, 1, 5)
        root.addWidget(filters_group)

        root.addWidget(self.status_label)
        root.addWidget(self.summary_label)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.table)
        splitter.addWidget(self.detail_box)
        splitter.setSizes([900, 420])
        root.addWidget(splitter, 1)

        exports_row = QHBoxLayout()
        export_filtered_button = QPushButton("Export filtered table")
        export_filtered_button.setObjectName("secondary")
        export_filtered_button.clicked.connect(self.export_filtered_table)
        exports_row.addWidget(export_filtered_button)
        export_unmatched_button = QPushButton("Export unmatched by target")
        export_unmatched_button.setObjectName("secondary")
        export_unmatched_button.clicked.connect(self.export_unmatched_for_target)
        exports_row.addWidget(export_unmatched_button)
        exports_row.addStretch(1)
        root.addLayout(exports_row)

    def _wire_events(self) -> None:
        self.audit_tau_combo.currentIndexChanged.connect(self._recompute_matches)
        self.target_filter_combo.currentIndexChanged.connect(self.apply_filters)
        self.status_filter_combo.currentIndexChanged.connect(self.apply_filters)
        self.eta_positive_checkbox.toggled.connect(self.apply_filters)
        self.loss_positive_checkbox.toggled.connect(self.apply_filters)
        self.search_edit.textChanged.connect(self.apply_filters)
        self.table.itemSelectionChanged.connect(self._update_detail_panel)

    def _configure_table(self) -> None:
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(8, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(9, QHeaderView.Stretch)
        self.table.setColumnWidth(0, 280)
        self.table.setColumnWidth(1, 180)
        self.table.setColumnWidth(9, 260)

    def _set_empty_state(self) -> None:
        self.audit_tau_combo.clear()
        self.table.setRowCount(0)
        self.detail_box.setPlainText("Load a priority CSV to inspect per-target GLAD matches.")

    def _wrap_layout(self, layout: QHBoxLayout) -> QWidget:
        wrapper = QWidget()
        wrapper.setLayout(layout)
        return wrapper

    def _pick_priority_csv(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select priority CSV", self.priority_csv_edit.text().strip(), "CSV files (*.csv)")
        if path:
            self.priority_csv_edit.setText(path)

    def _pick_metadata_json(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select metadata JSON", self.metadata_json_edit.text().strip(), "JSON files (*.json)")
        if path:
            self.metadata_json_edit.setText(path)

    def _pick_output_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select output directory", self.output_dir_edit.text().strip())
        if path:
            self.output_dir_edit.setText(path)

    def load_matches(self) -> None:
        priority_csv = self.priority_csv_edit.text().strip()
        metadata_json = self.metadata_json_edit.text().strip() or None
        if not priority_csv:
            QMessageBox.warning(self, "Missing input", "Select `lcia_flow_priority.csv` first.")
            return
        try:
            dataset, _, _, manifest = build_priority_glad_rows(priority_csv, metadata_json, None)
        except PriorityAnalysisError as exc:
            QMessageBox.critical(self, "Load failed", str(exc))
            return
        except Exception as exc:
            QMessageBox.critical(self, "Load failed", str(exc))
            return

        self.dataset = dataset
        self.asset_manifest = manifest
        self.audit_tau_combo.blockSignals(True)
        self.audit_tau_combo.clear()
        for pair in dataset.tau_pairs:
            self.audit_tau_combo.addItem(pair.tau_label, pair.tau)
        self.audit_tau_combo.blockSignals(False)
        default_index = self.audit_tau_combo.findData(0.95)
        self.audit_tau_combo.setCurrentIndex(default_index if default_index >= 0 else 0)
        self.status_label.setText(
            f"Loaded {len(dataset.rows)} priority rows. Bundled GLAD targets: {', '.join(_TARGET_KEYS)}."
        )
        self._recompute_matches()

    def _recompute_matches(self) -> None:
        if self.dataset is None or self.audit_tau_combo.count() == 0:
            return
        tau = self.audit_tau_combo.currentData()
        metadata_json = self.metadata_json_edit.text().strip() or None
        try:
            dataset, pair, rows, manifest = build_priority_glad_rows(
                self.priority_csv_edit.text().strip(),
                metadata_json,
                float(tau) if tau is not None else None,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Matching failed", str(exc))
            return
        self.dataset = dataset
        self.selected_pair = pair
        self.row_results = rows
        self.asset_manifest = manifest
        self.apply_filters()

    def apply_filters(self) -> None:
        if self.selected_pair is None:
            self.filtered_results = []
            self.table.setRowCount(0)
            self.detail_box.setPlainText("Load a priority CSV to inspect per-target GLAD matches.")
            return
        target_key = str(self.target_filter_combo.currentData())
        status_mode = str(self.status_filter_combo.currentData())
        search_text = self.search_edit.text().strip().casefold()

        filtered: list[PriorityGladRowResult] = []
        for item in self.row_results:
            row = item.priority_row
            if self.eta_positive_checkbox.isChecked() and row.eta(self.selected_pair) <= 0:
                continue
            if self.loss_positive_checkbox.isChecked() and row.loss_max(self.selected_pair) <= 0:
                continue
            if search_text and search_text not in row.flow_name.casefold() and search_text not in row.flow_id.casefold():
                continue
            if target_key != "all":
                match = item.target_results[target_key]
                if status_mode == "matched" and match.match_status not in _MATCHED_STATUSES:
                    continue
                if status_mode == "unmatched" and match.match_status not in _UNRESOLVED_STATUSES:
                    continue
                if status_mode == "problem" and match.match_status not in {"ambiguous", "conflict"}:
                    continue
            else:
                statuses = [result.match_status for result in item.target_results.values()]
                if status_mode == "matched" and not all(status in _MATCHED_STATUSES for status in statuses):
                    continue
                if status_mode == "unmatched" and all(status in _MATCHED_STATUSES for status in statuses):
                    continue
                if status_mode == "problem" and not any(status in {"ambiguous", "conflict"} for status in statuses):
                    continue
            filtered.append(item)

        self.filtered_results = filtered
        self._populate_table()
        self._update_summary_label()
        self._update_detail_panel()

    def _populate_table(self) -> None:
        self.table.setRowCount(len(self.filtered_results))
        for row_index, item in enumerate(self.filtered_results):
            row = item.priority_row
            self.table.setItem(row_index, 0, QTableWidgetItem(row.flow_name))
            self.table.setItem(row_index, 1, QTableWidgetItem(row.flow_id))
            self.table.setItem(row_index, 2, QTableWidgetItem(format(row.eta(self.selected_pair), ".6g")))
            self.table.setItem(row_index, 3, QTableWidgetItem(format(row.loss_max(self.selected_pair), ".6g")))
            for offset, target_key in enumerate(_TARGET_KEYS, start=4):
                self.table.setItem(row_index, offset, QTableWidgetItem(item.target_results[target_key].match_status))
            matched_uuid, matched_name, candidate_count = self._summary_match_columns(item)
            self.table.setItem(row_index, 8, QTableWidgetItem(matched_uuid))
            self.table.setItem(row_index, 9, QTableWidgetItem(matched_name))
            self.table.setItem(row_index, 10, QTableWidgetItem(str(candidate_count)))
        if self.filtered_results:
            self.table.selectRow(0)

    def _summary_match_columns(self, item: PriorityGladRowResult) -> tuple[str, str, int]:
        target_key = str(self.target_filter_combo.currentData())
        if target_key != "all":
            match = item.target_results[target_key]
            return (
                match.matched_target_uuid,
                match.matched_target_flow_name,
                match.candidate_count,
            )
        for target in _TARGET_KEYS:
            match = item.target_results[target]
            if match.matched_target_uuid or match.matched_target_flow_name:
                return (
                    match.matched_target_uuid,
                    match.matched_target_flow_name,
                    match.candidate_count,
                )
        total_candidates = sum(result.candidate_count for result in item.target_results.values())
        return ("", "", total_candidates)

    def _update_summary_label(self) -> None:
        if self.selected_pair is None:
            self.summary_label.setText("No priority file loaded.")
            return
        total = len(self.row_results)
        filtered = len(self.filtered_results)
        matched_all = sum(
            1 for item in self.filtered_results if all(result.match_status in _MATCHED_STATUSES for result in item.target_results.values())
        )
        unresolved_any = sum(
            1 for item in self.filtered_results if any(result.match_status in _UNRESOLVED_STATUSES for result in item.target_results.values())
        )
        self.summary_label.setText(
            f"Selected tau {self.selected_pair.tau_label} | showing {filtered} of {total} rows | "
            f"matched in all four targets: {matched_all} | unresolved in at least one target: {unresolved_any}"
        )

    def _update_detail_panel(self) -> None:
        if self.selected_pair is None:
            self.detail_box.setPlainText("Load a priority CSV to inspect per-target GLAD matches.")
            return
        selected = self.table.currentRow()
        if selected < 0 or selected >= len(self.filtered_results):
            self.detail_box.setPlainText("Select a row to inspect per-target matching details.")
            return
        item = self.filtered_results[selected]
        row = item.priority_row
        payload: dict[str, object] = {
            "flow_id": row.flow_id,
            "flow_name": row.flow_name,
            "compartment": row.compartment,
            "subcompartment": row.subcompartment,
            "reference_unit": row.reference_unit,
            "selected_audit_tau": self.selected_pair.tau_label,
            "eta_tau": row.eta(self.selected_pair),
            "loss_max_tau": row.loss_max(self.selected_pair),
            "matches": {},
        }
        for target_key in _TARGET_KEYS:
            match = item.target_results[target_key]
            payload["matches"][target_key] = {
                "match_status": match.match_status,
                "match_rule": match.match_rule,
                "matched_target_uuid": match.matched_target_uuid,
                "matched_target_flow_name": match.matched_target_flow_name,
                "matched_target_context": match.matched_target_context,
                "matched_target_unit": match.matched_target_unit,
                "candidate_count": match.candidate_count,
                "notes": match.notes,
                "candidate_rows": [
                    {
                        "flow_uuid": candidate.flow_uuid,
                        "flow_name": candidate.flow_name,
                        "context": candidate.context,
                        "unit": candidate.unit,
                    }
                    for candidate in match.candidates
                ],
            }
        self.detail_box.setPlainText(json.dumps(payload, indent=2, ensure_ascii=False))

    def export_all_reports(self) -> None:
        priority_csv = self.priority_csv_edit.text().strip()
        output_dir = self.output_dir_edit.text().strip()
        if not priority_csv:
            QMessageBox.warning(self, "Missing input", "Select `lcia_flow_priority.csv` first.")
            return
        if not output_dir:
            QMessageBox.warning(self, "Missing output", "Select an output folder first.")
            return
        try:
            write_priority_glad_outputs(
                priority_csv=priority_csv,
                output_dir=output_dir,
                metadata_json=self.metadata_json_edit.text().strip() or None,
                audit_tau=self.selected_pair.tau if self.selected_pair is not None else None,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
            return
        QMessageBox.information(self, "Reports written", f"Priority -> GLAD reports written to:\n{output_dir}")

    def export_filtered_table(self) -> None:
        if self.selected_pair is None or not self.filtered_results:
            QMessageBox.warning(self, "Nothing to export", "Load a priority CSV and keep at least one visible row.")
            return
        default_path = Path(self.priority_csv_edit.text().strip()).with_name("priority_glad_filtered.csv")
        path, _ = QFileDialog.getSaveFileName(self, "Export filtered table", str(default_path), "CSV files (*.csv)")
        if not path:
            return
        fieldnames = [self.table.horizontalHeaderItem(index).text() for index in range(self.table.columnCount())]
        rows = [self._serialise_visible_table_row(item) for item in self.filtered_results]
        write_manifest_csv(Path(path), rows, fieldnames)

    def export_unmatched_for_target(self) -> None:
        if self.selected_pair is None or not self.row_results:
            QMessageBox.warning(self, "Nothing to export", "Load a priority CSV first.")
            return
        target_key = str(self.target_filter_combo.currentData())
        if target_key == "all":
            QMessageBox.warning(self, "Select target", "Choose a specific target before exporting unmatched rows.")
            return
        default_path = Path(self.priority_csv_edit.text().strip()).with_name(f"{target_key}_unmatched_filtered.csv")
        path, _ = QFileDialog.getSaveFileName(self, "Export unmatched rows", str(default_path), "CSV files (*.csv)")
        if not path:
            return
        unresolved = [item for item in self.filtered_results if item.target_results[target_key].match_status in _UNRESOLVED_STATUSES]
        rows = [self._serialise_for_export(item, target_key=target_key) for item in unresolved]
        write_manifest_csv(Path(path), rows, priority_glad_output_columns(self.selected_pair))

    def _serialise_for_export(self, item: PriorityGladRowResult, *, target_key: str | None = None) -> dict[str, object]:
        export_target = target_key or str(self.target_filter_combo.currentData())
        if export_target == "all":
            export_target = _TARGET_KEYS[0]
        return serialise_priority_glad_row(item, item.target_results[export_target])

    def _serialise_visible_table_row(self, item: PriorityGladRowResult) -> dict[str, object]:
        row = item.priority_row
        matched_uuid, matched_name, candidate_count = self._summary_match_columns(item)
        return {
            "flow_name": row.flow_name,
            "flow_id": row.flow_id,
            "eta_tau": row.eta(self.selected_pair),
            "loss_max_tau": row.loss_max(self.selected_pair),
            "ecoinventEFv3.7": item.target_results["ecoinventEFv3.7"].match_status,
            "ILCD_EFv3.0": item.target_results["ILCD_EFv3.0"].match_status,
            "FEDEFLv1.0.3": item.target_results["FEDEFLv1.0.3"].match_status,
            "IDEA_EFv2.3": item.target_results["IDEA_EFv2.3"].match_status,
            "matched_target_uuid": matched_uuid,
            "matched_target_name": matched_name,
            "candidate_count": candidate_count,
        }
