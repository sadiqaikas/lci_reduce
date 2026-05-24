"""Analyse compact LCIA flow-priority CSV files."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from .errors import LciReduceError
from .manifest import write_manifest_csv


BASE_PRIORITY_COLUMNS = [
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
    "cf_status",
]

RANKED_EXPORT_COLUMNS = [
    "rank",
    "flow_id",
    "flow_name",
    "compartment",
    "subcompartment",
    "reference_unit",
    "eta_tau",
    "loss_max_tau",
    "tau_entry_min",
    "tau_entry_median",
    "tau_entry_max",
    "occurrence_count",
    "characterised_occurrence_count",
    "cf_status",
]

_ETA_COLUMN_RE = re.compile(r"^eta_(?P<token>[0-9A-Za-z_]+)$")
_EPSILON = 1e-12


class PriorityAnalysisError(LciReduceError):
    """Raised when a priority CSV cannot be analysed safely."""


@dataclass(frozen=True)
class TauColumnPair:
    tau: float
    token: str
    eta_column: str
    loss_max_column: str

    @property
    def tau_label(self) -> str:
        return format(float(self.tau), ".15g")

    def to_dict(self) -> dict[str, Any]:
        return {
            "tau": self.tau,
            "token": self.token,
            "eta_column": self.eta_column,
            "loss_max_column": self.loss_max_column,
        }


@dataclass
class PriorityRecord:
    row_number: int
    original_order: int
    raw: dict[str, str]
    flow_id: str
    flow_name: str
    compartment: str
    subcompartment: str
    reference_unit: str
    occurrence_count: int
    characterised_occurrence_count: int
    tau_entry_min: float | None
    tau_entry_median: float | None
    tau_entry_max: float | None
    cf_status: str
    metrics: dict[str, tuple[float, float]]

    def eta(self, pair: TauColumnPair) -> float:
        return self.metrics[pair.token][0]

    def loss_max(self, pair: TauColumnPair) -> float:
        return self.metrics[pair.token][1]


@dataclass
class PriorityDataset:
    source_path: str
    fieldnames: list[str]
    rows: list[PriorityRecord]
    tau_pairs: list[TauColumnPair]
    metadata_path: str | None = None
    metadata: dict[str, Any] | None = None
    rows_by_flow_id: dict[str, PriorityRecord] = field(init=False, default_factory=dict)
    rows_by_exact_name: dict[str, list[PriorityRecord]] = field(init=False, default_factory=dict)
    rows_by_folded_name: dict[str, list[PriorityRecord]] = field(init=False, default_factory=dict)
    tau_pairs_by_key: dict[str, TauColumnPair] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self.rows_by_flow_id = {row.flow_id: row for row in self.rows}
        for row in self.rows:
            self.rows_by_exact_name.setdefault(row.flow_name, []).append(row)
            self.rows_by_folded_name.setdefault(row.flow_name.casefold(), []).append(row)
        self.tau_pairs_by_key = {
            _tau_key(pair.tau): pair
            for pair in self.tau_pairs
        }

    def get_tau_pair(self, tau: float | None = None) -> TauColumnPair:
        if not self.tau_pairs:
            raise PriorityAnalysisError("No audit tau columns were detected in the priority CSV.")
        if tau is None:
            preferred = self.tau_pairs_by_key.get(_tau_key(0.95))
            return preferred if preferred is not None else self.tau_pairs[0]
        pair = self.tau_pairs_by_key.get(_tau_key(tau))
        if pair is None:
            available = ", ".join(item.tau_label for item in self.tau_pairs)
            raise PriorityAnalysisError(
                f"Audit tau {format(float(tau), '.15g')} is not available in this priority CSV. "
                f"Detected tau values: {available}."
            )
        return pair

    def has_tau(self, tau: float) -> bool:
        return _tau_key(tau) in self.tau_pairs_by_key


@dataclass
class SelectionMatchResult:
    matched_rows: list[PriorityRecord]
    matched_flow_ids: list[str]
    unmatched_items: list[str]
    ambiguous_items: dict[str, list[str]]

    def to_dict(self, pair: TauColumnPair | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "matched_count": len(self.matched_rows),
            "matched_flow_ids": list(self.matched_flow_ids),
            "unmatched_items": list(self.unmatched_items),
            "ambiguous_items": dict(self.ambiguous_items),
        }
        if pair is not None:
            payload["matched_flows"] = [serialise_row(row, pair) for row in self.matched_rows]
        return payload


@dataclass
class GroupBoundResult:
    pair: TauColumnPair
    selected_count: int
    lower_bound: float | None
    sum_loss_max: float
    upper_bound: float | None
    capped: bool
    exact_eta: float | None
    exact_reason: str | None
    interval_text: str
    interpretation: list[str]
    ranked_by_eta: list[PriorityRecord]
    ranked_by_loss_max: list[PriorityRecord]

    def to_dict(self, top_n: int = 20) -> dict[str, Any]:
        return {
            "selected_count": self.selected_count,
            "tau": self.pair.tau,
            "lower_bound_eta": self.lower_bound,
            "sum_loss_max": self.sum_loss_max,
            "upper_bound_eta": self.upper_bound,
            "capped_at_tau": self.capped,
            "exact_eta": self.exact_eta,
            "exact_reason": self.exact_reason,
            "interval": self.interval_text,
            "interpretation": list(self.interpretation),
            "top_selected_by_eta": [
                serialise_row(row, self.pair, rank=index)
                for index, row in enumerate(self.ranked_by_eta[:top_n], start=1)
            ],
            "top_selected_by_loss_max": [
                serialise_row(row, self.pair, rank=index)
                for index, row in enumerate(self.ranked_by_loss_max[:top_n], start=1)
            ],
        }


def load_priority_dataset(
    priority_csv: str | Path,
    metadata_json: str | Path | None = None,
) -> PriorityDataset:
    csv_path = Path(priority_csv)
    if not csv_path.exists():
        raise PriorityAnalysisError(f"Priority CSV not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        if not fieldnames:
            raise PriorityAnalysisError("Priority CSV is empty or missing a header row.")
        _validate_priority_fieldnames(fieldnames)
        tau_pairs = detect_tau_column_pairs(fieldnames)
        rows: list[PriorityRecord] = []
        seen_flow_ids: set[str] = set()
        for original_order, raw_row in enumerate(reader):
            cleaned = {
                field: ((raw_row.get(field) if raw_row is not None else "") or "").strip()
                for field in fieldnames
            }
            if not any(cleaned.values()):
                continue
            row_number = original_order + 2
            flow_id = cleaned["flow_id"]
            if not flow_id:
                raise PriorityAnalysisError(f"Row {row_number} is missing flow_id.")
            if flow_id in seen_flow_ids:
                raise PriorityAnalysisError(f"Priority CSV contains duplicate flow_id values: {flow_id}")
            seen_flow_ids.add(flow_id)

            metrics: dict[str, tuple[float, float]] = {}
            for pair in tau_pairs:
                eta = _parse_required_float(cleaned[pair.eta_column], row_number, pair.eta_column)
                loss_max = _parse_required_float(cleaned[pair.loss_max_column], row_number, pair.loss_max_column)
                if eta < -_EPSILON:
                    raise PriorityAnalysisError(
                        f"Row {row_number} has a negative single-flow shortfall in {pair.eta_column}."
                    )
                if loss_max < -_EPSILON:
                    raise PriorityAnalysisError(
                        f"Row {row_number} has a negative raw coverage loss in {pair.loss_max_column}."
                    )
                metrics[pair.token] = (max(eta, 0.0), max(loss_max, 0.0))

            rows.append(
                PriorityRecord(
                    row_number=row_number,
                    original_order=original_order,
                    raw=cleaned,
                    flow_id=flow_id,
                    flow_name=cleaned["flow_name"],
                    compartment=cleaned["compartment"],
                    subcompartment=cleaned["subcompartment"],
                    reference_unit=cleaned["reference_unit"],
                    occurrence_count=_parse_int(cleaned["occurrence_count"], row_number, "occurrence_count"),
                    characterised_occurrence_count=_parse_int(
                        cleaned["characterised_occurrence_count"],
                        row_number,
                        "characterised_occurrence_count",
                    ),
                    tau_entry_min=_parse_optional_float(cleaned["tau_entry_min"], row_number, "tau_entry_min"),
                    tau_entry_median=_parse_optional_float(
                        cleaned["tau_entry_median"],
                        row_number,
                        "tau_entry_median",
                    ),
                    tau_entry_max=_parse_optional_float(cleaned["tau_entry_max"], row_number, "tau_entry_max"),
                    cf_status=cleaned["cf_status"],
                    metrics=metrics,
                )
            )

    metadata_payload = load_priority_metadata(metadata_json) if metadata_json else None
    return PriorityDataset(
        source_path=str(csv_path),
        fieldnames=fieldnames,
        rows=rows,
        tau_pairs=tau_pairs,
        metadata_path=str(metadata_json) if metadata_json else None,
        metadata=metadata_payload,
    )


def load_priority_metadata(metadata_json: str | Path) -> dict[str, Any]:
    metadata_path = Path(metadata_json)
    if not metadata_path.exists():
        raise PriorityAnalysisError(f"Metadata JSON not found: {metadata_path}")
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PriorityAnalysisError(f"Metadata JSON is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise PriorityAnalysisError("Metadata JSON must contain a single JSON object.")
    return payload


def detect_tau_column_pairs(fieldnames: Sequence[str]) -> list[TauColumnPair]:
    eta_tokens: dict[str, str] = {}
    loss_tokens: set[str] = set()
    for field in fieldnames:
        match = _ETA_COLUMN_RE.match(field)
        if match:
            eta_tokens[match.group("token")] = field
        elif field.startswith("loss_max_"):
            loss_tokens.add(field[len("loss_max_"):])

    all_tokens = sorted(set(eta_tokens) | loss_tokens)
    if not all_tokens:
        raise PriorityAnalysisError(
            "Priority CSV is missing eta_<tau> / loss_max_<tau> columns. "
            "At least one audit tau column pair is required."
        )

    pairs: list[TauColumnPair] = []
    missing_pairs: list[str] = []
    for token in all_tokens:
        eta_column = eta_tokens.get(token)
        loss_max_column = f"loss_max_{token}" if token in loss_tokens else None
        if eta_column is None or loss_max_column is None:
            missing_pairs.append(token)
            continue
        tau = _tau_from_token(token)
        pairs.append(
            TauColumnPair(
                tau=tau,
                token=token,
                eta_column=eta_column,
                loss_max_column=loss_max_column,
            )
        )
    if missing_pairs:
        formatted = ", ".join(sorted(missing_pairs))
        raise PriorityAnalysisError(
            f"Priority CSV is missing one side of an eta/loss_max pair for tau token(s): {formatted}"
        )
    return sorted(pairs, key=lambda item: (item.tau, item.token))


def build_priority_overview(dataset: PriorityDataset) -> dict[str, Any]:
    overview = {
        "total_flows": len(dataset.rows),
        "characterised_flows": sum(row.cf_status == "characterised" for row in dataset.rows),
        "partly_characterised_flows": sum(row.cf_status == "partly_characterised" for row in dataset.rows),
        "uncharacterised_flows": sum(row.cf_status == "uncharacterised" for row in dataset.rows),
        "tau_metrics": {},
    }
    for pair in dataset.tau_pairs:
        overview["tau_metrics"][pair.token] = {
            "tau": pair.tau,
            "eta_positive_count": sum(row.eta(pair) > _EPSILON for row in dataset.rows),
            "loss_positive_count": sum(row.loss_max(pair) > _EPSILON for row in dataset.rows),
        }
    return overview


def filter_priority_rows(
    dataset: PriorityDataset,
    pair: TauColumnPair,
    *,
    search_text: str = "",
    cf_status: str = "all",
    eta_positive_only: bool = False,
    loss_positive_only: bool = False,
    critical_only_at_099: bool = False,
    group_risk_only: bool = False,
) -> list[PriorityRecord]:
    search = search_text.strip().casefold()
    pair_095 = dataset.get_tau_pair(0.95) if dataset.has_tau(0.95) else None
    pair_099 = dataset.get_tau_pair(0.99) if dataset.has_tau(0.99) else None
    rows: list[PriorityRecord] = []
    for row in dataset.rows:
        if search and search not in row.flow_id.casefold() and search not in row.flow_name.casefold():
            continue
        if cf_status != "all" and row.cf_status != cf_status:
            continue
        if eta_positive_only and row.eta(pair) <= _EPSILON:
            continue
        if loss_positive_only and row.loss_max(pair) <= _EPSILON:
            continue
        if critical_only_at_099:
            if pair_095 is None or pair_099 is None:
                raise PriorityAnalysisError(
                    "The critical-only-at-0.99 filter requires both 0.95 and 0.99 audit tau columns."
                )
            if not (row.eta(pair_095) <= _EPSILON and row.eta(pair_099) > _EPSILON):
                continue
        if group_risk_only and not (row.eta(pair) <= _EPSILON and row.loss_max(pair) > _EPSILON):
            continue
        rows.append(row)
    return rows


def rank_rows_by_eta(rows: Sequence[PriorityRecord], pair: TauColumnPair) -> list[PriorityRecord]:
    return sorted(rows, key=lambda row: _eta_sort_key(row, pair))


def rank_rows_by_loss_max(rows: Sequence[PriorityRecord], pair: TauColumnPair) -> list[PriorityRecord]:
    return sorted(rows, key=lambda row: _loss_sort_key(row, pair))


def match_flow_ids(dataset: PriorityDataset, flow_ids: Sequence[str]) -> SelectionMatchResult:
    matched_rows: list[PriorityRecord] = []
    matched_flow_ids: list[str] = []
    unmatched_items: list[str] = []
    seen_flow_ids: set[str] = set()
    for item in flow_ids:
        token = item.strip()
        if not token:
            continue
        row = dataset.rows_by_flow_id.get(token)
        if row is None:
            unmatched_items.append(token)
            continue
        if row.flow_id not in seen_flow_ids:
            seen_flow_ids.add(row.flow_id)
            matched_rows.append(row)
            matched_flow_ids.append(row.flow_id)
    return SelectionMatchResult(
        matched_rows=matched_rows,
        matched_flow_ids=matched_flow_ids,
        unmatched_items=unmatched_items,
        ambiguous_items={},
    )


def match_flow_names(dataset: PriorityDataset, names: Sequence[str]) -> SelectionMatchResult:
    matched_rows: list[PriorityRecord] = []
    matched_flow_ids: list[str] = []
    unmatched_items: list[str] = []
    ambiguous_items: dict[str, list[str]] = {}
    seen_flow_ids: set[str] = set()
    for item in names:
        token = item.strip()
        if not token:
            continue
        exact = dataset.rows_by_exact_name.get(token, [])
        if len(exact) == 1:
            row = exact[0]
            if row.flow_id not in seen_flow_ids:
                seen_flow_ids.add(row.flow_id)
                matched_rows.append(row)
                matched_flow_ids.append(row.flow_id)
            continue
        if len(exact) > 1:
            ambiguous_items[token] = [_candidate_text(row) for row in exact]
            continue
        folded = dataset.rows_by_folded_name.get(token.casefold(), [])
        if len(folded) == 1:
            row = folded[0]
            if row.flow_id not in seen_flow_ids:
                seen_flow_ids.add(row.flow_id)
                matched_rows.append(row)
                matched_flow_ids.append(row.flow_id)
            continue
        if len(folded) > 1:
            ambiguous_items[token] = [_candidate_text(row) for row in folded]
            continue
        unmatched_items.append(token)
    return SelectionMatchResult(
        matched_rows=matched_rows,
        matched_flow_ids=matched_flow_ids,
        unmatched_items=unmatched_items,
        ambiguous_items=ambiguous_items,
    )


def match_mixed_flow_items(dataset: PriorityDataset, items: Sequence[str]) -> SelectionMatchResult:
    matched_rows: list[PriorityRecord] = []
    matched_flow_ids: list[str] = []
    unmatched_items: list[str] = []
    ambiguous_items: dict[str, list[str]] = {}
    seen_flow_ids: set[str] = set()
    for item in items:
        token = item.strip()
        if not token:
            continue
        row = dataset.rows_by_flow_id.get(token)
        if row is not None:
            if row.flow_id not in seen_flow_ids:
                seen_flow_ids.add(row.flow_id)
                matched_rows.append(row)
                matched_flow_ids.append(row.flow_id)
            continue
        exact = dataset.rows_by_exact_name.get(token, [])
        if len(exact) == 1:
            row = exact[0]
            if row.flow_id not in seen_flow_ids:
                seen_flow_ids.add(row.flow_id)
                matched_rows.append(row)
                matched_flow_ids.append(row.flow_id)
            continue
        if len(exact) > 1:
            ambiguous_items[token] = [_candidate_text(row) for row in exact]
            continue
        folded = dataset.rows_by_folded_name.get(token.casefold(), [])
        if len(folded) == 1:
            row = folded[0]
            if row.flow_id not in seen_flow_ids:
                seen_flow_ids.add(row.flow_id)
                matched_rows.append(row)
                matched_flow_ids.append(row.flow_id)
            continue
        if len(folded) > 1:
            ambiguous_items[token] = [_candidate_text(row) for row in folded]
            continue
        unmatched_items.append(token)
    return SelectionMatchResult(
        matched_rows=matched_rows,
        matched_flow_ids=matched_flow_ids,
        unmatched_items=unmatched_items,
        ambiguous_items=ambiguous_items,
    )


def analyse_selected_group(
    rows: Sequence[PriorityRecord],
    pair: TauColumnPair,
) -> GroupBoundResult:
    ranked_by_eta = rank_rows_by_eta(rows, pair)
    ranked_by_loss_max = rank_rows_by_loss_max(rows, pair)
    if not rows:
        return GroupBoundResult(
            pair=pair,
            selected_count=0,
            lower_bound=None,
            sum_loss_max=0.0,
            upper_bound=None,
            capped=False,
            exact_eta=None,
            exact_reason=None,
            interval_text="No flows selected.",
            interpretation=["No flows selected."],
            ranked_by_eta=ranked_by_eta,
            ranked_by_loss_max=ranked_by_loss_max,
        )

    lower_bound = max(row.eta(pair) for row in rows)
    sum_loss_max = sum(row.loss_max(pair) for row in rows)
    capped = sum_loss_max > pair.tau + _EPSILON
    upper_bound = min(pair.tau, sum_loss_max)
    exact_eta: float | None = None
    exact_reason: str | None = None

    if len(rows) == 1:
        exact_eta = rows[0].eta(pair)
        exact_reason = "Single selected flow: eta_F(tau) equals that flow's eta value from the CSV."
    elif abs(lower_bound - upper_bound) <= _EPSILON:
        exact_eta = lower_bound
        exact_reason = "Lower and upper bounds coincide, so the selected-group eta is determined exactly."

    interpretation: list[str] = []
    if exact_eta is not None and exact_eta <= 0.02 + _EPSILON:
        interpretation.append("Low selected-flow consequence. Exact eta is below 2 percentage points.")
    elif upper_bound <= 0.02 + _EPSILON:
        interpretation.append(
            "Low compact-screen group risk. Even the conservative upper bound is below 2 percentage points."
        )
    if lower_bound > _EPSILON:
        interpretation.append("At least one selected flow individually breaks the certificate.")
    if lower_bound >= 0.10 - _EPSILON:
        interpretation.append("Definitely serious. The lower bound alone indicates a large certificate shortfall.")
    if lower_bound <= _EPSILON and upper_bound >= 0.10 - _EPSILON:
        interpretation.append(
            "No selected flow breaks the certificate alone, but accumulation risk may be substantial. "
            "Exact audit is recommended if this group is actually unresolved."
        )
    if capped:
        interpretation.append("Upper bound capped at tau.")
    if exact_reason is not None:
        interpretation.append(exact_reason)
    if not interpretation:
        interpretation.append("Compact-screen bound computed from the priority CSV. Review the numeric interval directly.")

    if exact_eta is not None:
        interval_text = f"Exact eta_F({pair.tau_label}) = {_format_metric(exact_eta)}."
    else:
        lower_text = _format_metric(lower_bound)
        upper_text = _format_metric(upper_bound)
        interval_text = (
            f"eta_F({pair.tau_label}) is bounded by [{lower_text}, {upper_text}]. "
            "Compact-screen bound from the CSV."
        )
    return GroupBoundResult(
        pair=pair,
        selected_count=len(rows),
        lower_bound=lower_bound,
        sum_loss_max=sum_loss_max,
        upper_bound=upper_bound,
        capped=capped,
        exact_eta=exact_eta,
        exact_reason=exact_reason,
        interval_text=interval_text,
        interpretation=interpretation,
        ranked_by_eta=ranked_by_eta,
        ranked_by_loss_max=ranked_by_loss_max,
    )


def serialise_row(
    row: PriorityRecord,
    pair: TauColumnPair,
    *,
    rank: int | None = None,
) -> dict[str, Any]:
    payload = {
        "flow_id": row.flow_id,
        "flow_name": row.flow_name,
        "compartment": row.compartment,
        "subcompartment": row.subcompartment,
        "reference_unit": row.reference_unit,
        "eta_tau": row.eta(pair),
        "loss_max_tau": row.loss_max(pair),
        "tau_entry_min": row.tau_entry_min,
        "tau_entry_median": row.tau_entry_median,
        "tau_entry_max": row.tau_entry_max,
        "occurrence_count": row.occurrence_count,
        "characterised_occurrence_count": row.characterised_occurrence_count,
        "cf_status": row.cf_status,
        "selected_audit_tau": pair.tau,
        "eta_column": pair.eta_column,
        "loss_max_column": pair.loss_max_column,
    }
    if rank is not None:
        payload["rank"] = rank
    return payload


def build_priority_summary(
    dataset: PriorityDataset,
    pair: TauColumnPair,
    *,
    top_n: int = 20,
    filtered_rows: Sequence[PriorityRecord] | None = None,
    selected_rows: Sequence[PriorityRecord] | None = None,
    selection_match: SelectionMatchResult | None = None,
) -> dict[str, Any]:
    filtered = list(filtered_rows if filtered_rows is not None else dataset.rows)
    ranked_by_eta = rank_rows_by_eta(filtered, pair)
    ranked_by_loss_max = rank_rows_by_loss_max(filtered, pair)
    top_n = max(int(top_n), 1)
    summary: dict[str, Any] = {
        "priority_csv": dataset.source_path,
        "metadata_json": dataset.metadata_path or "",
        "schema_validation": {
            "status": "ok",
            "required_base_columns": list(BASE_PRIORITY_COLUMNS),
            "detected_tau_values": [item.tau for item in dataset.tau_pairs],
            "detected_tau_columns": [item.to_dict() for item in dataset.tau_pairs],
            "row_count": len(dataset.rows),
        },
        "metadata_summary": _metadata_summary(dataset.metadata),
        "overview": build_priority_overview(dataset),
        "selected_audit_tau": pair.tau,
        "selected_columns": {
            "eta_column": pair.eta_column,
            "loss_max_column": pair.loss_max_column,
        },
        "filtered_row_count": len(filtered),
        "top_n": top_n,
        "top_by_eta": [
            serialise_row(row, pair, rank=index)
            for index, row in enumerate(ranked_by_eta[:top_n], start=1)
        ],
        "top_by_loss_max": [
            serialise_row(row, pair, rank=index)
            for index, row in enumerate(ranked_by_loss_max[:top_n], start=1)
        ],
    }
    if selection_match is not None:
        summary["selection_match"] = selection_match.to_dict(pair)
    if selected_rows is not None:
        group_result = analyse_selected_group(selected_rows, pair)
        summary["selected_flow_analysis"] = group_result.to_dict(top_n=top_n)
    return summary


def write_ranked_csv(
    output_path: str | Path,
    rows: Sequence[PriorityRecord],
    pair: TauColumnPair,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        serialise_ranked_csv_row(row, pair, rank=index)
        for index, row in enumerate(rank_rows_by_eta(rows, pair), start=1)
    ]
    write_manifest_csv(path, payload, RANKED_EXPORT_COLUMNS)


def write_selected_rows_csv(
    output_path: str | Path,
    rows: Sequence[PriorityRecord],
    pair: TauColumnPair,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        serialise_ranked_csv_row(row, pair, rank=index)
        for index, row in enumerate(rank_rows_by_eta(rows, pair), start=1)
    ]
    write_manifest_csv(path, payload, RANKED_EXPORT_COLUMNS)


def write_summary_json(output_path: str | Path, summary: dict[str, Any]) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")


def parse_repeated_option_items(values: Sequence[str] | None) -> list[str]:
    items: list[str] = []
    for value in values or []:
        text = value.strip()
        if not text:
            continue
        try:
            parsed = next(csv.reader([text], skipinitialspace=True))
        except Exception:
            parsed = text.split(",")
        for item in parsed:
            candidate = item.strip()
            if candidate:
                items.append(candidate)
    return items


def parse_multiline_items(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def serialise_ranked_csv_row(
    row: PriorityRecord,
    pair: TauColumnPair,
    *,
    rank: int,
) -> dict[str, Any]:
    payload = serialise_row(row, pair, rank=rank)
    return {column: payload.get(column, "") for column in RANKED_EXPORT_COLUMNS}


def _validate_priority_fieldnames(fieldnames: Sequence[str]) -> None:
    duplicates = sorted({field for field in fieldnames if fieldnames.count(field) > 1})
    if duplicates:
        raise PriorityAnalysisError(
            f"Priority CSV header contains duplicate columns: {', '.join(duplicates)}"
        )
    missing = [column for column in BASE_PRIORITY_COLUMNS if column not in fieldnames]
    if missing:
        raise PriorityAnalysisError(
            "Priority CSV is missing required base columns: " + ", ".join(missing)
        )


def _parse_int(value: str, row_number: int, column: str) -> int:
    if value == "":
        raise PriorityAnalysisError(f"Row {row_number} is missing required integer column {column}.")
    try:
        parsed = float(value)
    except ValueError as exc:
        raise PriorityAnalysisError(
            f"Row {row_number} has an invalid integer in {column}: {value}"
        ) from exc
    if not parsed.is_integer():
        raise PriorityAnalysisError(
            f"Row {row_number} has a non-integer value in {column}: {value}"
        )
    return int(parsed)


def _parse_optional_float(value: str, row_number: int, column: str) -> float | None:
    if value == "":
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise PriorityAnalysisError(
            f"Row {row_number} has an invalid numeric value in {column}: {value}"
        ) from exc


def _parse_required_float(value: str, row_number: int, column: str) -> float:
    parsed = _parse_optional_float(value, row_number, column)
    if parsed is None:
        raise PriorityAnalysisError(f"Row {row_number} is missing required numeric column {column}.")
    return parsed


def _tau_from_token(token: str) -> float:
    try:
        tau = float(token.replace("_", "."))
    except ValueError as exc:
        raise PriorityAnalysisError(
            f"Could not parse audit tau token from column suffix: {token}"
        ) from exc
    if tau <= 0 or tau > 1:
        raise PriorityAnalysisError(f"Audit tau values must be in (0, 1]; got {tau}")
    return tau


def _tau_key(tau: float) -> str:
    return format(float(tau), ".15g")


def _metadata_summary(metadata: dict[str, Any] | None) -> dict[str, Any] | None:
    if not metadata:
        return None
    selected_categories = metadata.get("selected_impact_categories")
    return {
        "file_type": metadata.get("file_type", ""),
        "database_name": metadata.get("database_name", ""),
        "audit_tau_values": metadata.get("audit_tau_values", []),
        "selected_impact_category_count": len(selected_categories) if isinstance(selected_categories, list) else None,
        "algorithm": metadata.get("algorithm", ""),
    }


def _eta_sort_key(row: PriorityRecord, pair: TauColumnPair) -> tuple[Any, ...]:
    return (
        -row.eta(pair),
        -row.loss_max(pair),
        row.tau_entry_min is None,
        float("inf") if row.tau_entry_min is None else row.tau_entry_min,
        row.tau_entry_median is None,
        float("inf") if row.tau_entry_median is None else row.tau_entry_median,
        -row.characterised_occurrence_count,
        row.flow_id,
        row.original_order,
    )


def _loss_sort_key(row: PriorityRecord, pair: TauColumnPair) -> tuple[Any, ...]:
    return (
        -row.loss_max(pair),
        -row.eta(pair),
        row.tau_entry_min is None,
        float("inf") if row.tau_entry_min is None else row.tau_entry_min,
        row.tau_entry_median is None,
        float("inf") if row.tau_entry_median is None else row.tau_entry_median,
        -row.characterised_occurrence_count,
        row.flow_id,
        row.original_order,
    )


def _candidate_text(row: PriorityRecord) -> str:
    parts = [row.flow_id, row.flow_name]
    location = "/".join(part for part in [row.compartment, row.subcompartment] if part)
    if location:
        parts.append(location)
    return " | ".join(parts)


def _format_metric(value: float | None) -> str:
    if value is None:
        return "-"
    return format(float(value), ".6g")
