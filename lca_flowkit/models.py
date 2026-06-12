"""Format-neutral data models used by the clean package.

These dataclasses are part of the audit story.  They show, explicitly, which
pieces of information the package treats as scientifically relevant:

- parsed archive entries, so output writers can preserve source content;
- flow, exchange, and unit identities, so matching remains inspectable;
- factor candidates, so ambiguity is represented rather than hidden;
- scenario rows, so coverage guarantees can be tied back to LCIA contexts;
- workflow result objects, so notebook users receive structured outputs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable


ProgressCallback = Callable[["ProgressEvent"], None]


@dataclass(frozen=True)
class Location:
    """Normalised location identity used across exchanges, flows, and factors.

    The package stores several parallel location hints because source datasets
    are rarely consistent about which field is authoritative.  Scenario
    matching later uses all non-empty tokens conservatively.
    """
    location_id: str = ""
    name: str = ""
    region: str = ""

    def tokens(self) -> tuple[str, ...]:
        """Return de-duplicated location tokens for exact/fallback matching.

        The returned tuple preserves first-seen order so later diagnostics stay
        stable and human-readable.
        """
        values: list[str] = []
        for raw in (self.location_id, self.name, self.region):
            token = str(raw or "").strip()
            if token and token not in values:
                values.append(token)
        return tuple(values)


@dataclass(frozen=True)
class UnitDefinition:
    """Unit metadata needed for strict compatibility and conversion checks.

    Only the fields needed for auditably proving unit compatibility are stored
    here.  The package does not attempt a general symbolic unit algebra system.
    """
    unit_id: str
    name: str
    group_id: str = ""
    group_name: str = ""
    conversion_factor: float | None = None
    is_reference_unit: bool = False
    flow_property_id: str = ""
    flow_property_name: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FlowDefinition:
    """Format-neutral description of one product, waste, or elementary flow.

    ``is_elementary`` is especially important because only elementary exchanges
    are eligible for removal in the reducer.
    """
    flow_id: str
    name: str
    flow_type: str
    compartment: str = ""
    subcompartment: str = ""
    is_elementary: bool = False
    location: Location = field(default_factory=Location)
    reference_flow_property_id: str = ""
    reference_flow_property_name: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def category_path(self) -> str:
        """Reconstruct the full category path from split compartment fields."""
        if self.compartment and self.subcompartment:
            return f"{self.compartment}/{self.subcompartment}"
        return self.compartment or self.subcompartment


@dataclass(frozen=True)
class Exchange:
    """One process exchange occurrence after format-specific parsing.

    The reducer works on exchange *occurrences*, not only on flow identities.
    Two exchanges in the same process that reference the same flow still remain
    separate columns if they occur separately in the process exchange list.
    """
    exchange_id: str
    index: int
    amount: float
    flow_id: str
    flow_name: str
    unit_name: str = ""
    unit_id: str = ""
    flow_property_id: str = ""
    flow_property_name: str = ""
    location: Location = field(default_factory=Location)
    provider_id: str = ""
    is_quantitative_reference: bool = False
    raw: Any = None


@dataclass
class ProcessRecord:
    """One parsed process with typed exchanges and original raw payload.

    ``raw`` is kept so writers can rebuild reduced outputs without reparsing the
    source archive from disk.
    """
    process_id: str
    name: str
    source_path: str
    location: Location
    exchanges: list[Exchange]
    raw: Any


@dataclass(frozen=True)
class ImpactFactorCandidate:
    """One admissible LCIA factor candidate before ambiguity resolution.

    A candidate here is intentionally *pre-resolution*.  The scenario builder
    is responsible for deciding whether several candidates collapse to one exact
    effect or must remain as explicit ambiguity.
    """
    category_id: str
    category_name: str
    method_id: str
    method_name: str
    flow_id: str
    flow_name: str
    value: float
    unit_name: str = ""
    unit_id: str = ""
    flow_property_id: str = ""
    flow_property_name: str = ""
    location: Location = field(default_factory=Location)
    compartment: str = ""
    subcompartment: str = ""
    coverage_tokens: tuple[str, ...] = field(default_factory=tuple)
    source_path: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class ImpactCategory:
    """One impact category plus every factor candidate grouped by flow ID.

    ``factors_by_flow`` preserves all candidates grouped under the flow identity
    they may characterise.  The grouping is convenient for contribution
    construction but does not imply uniqueness.
    """
    category_id: str
    name: str
    method_id: str
    method_name: str
    reference_unit: str = ""
    path: str = ""
    source_path: str = ""
    method_source_path: str = ""
    factors_by_flow: dict[str, list[ImpactFactorCandidate]] = field(default_factory=dict)


@dataclass
class ArchiveEntry:
    """One raw file entry from a source archive or folder."""
    path: str
    raw_bytes: bytes
    kind: str
    object_type: str
    object_id: str
    name: str
    parsed: Any = None


@dataclass
class DatasetBundle:
    """All parsed objects needed from one database or methods source.

    This bundle is the hand-off point between file-format parsing and the
    scientific workflows.
    """
    source_path: str
    source_name: str
    source_format: str
    entries: list[ArchiveEntry]
    processes: list[ProcessRecord]
    flows: dict[str, FlowDefinition]
    units: dict[str, UnitDefinition]
    categories: dict[str, ImpactCategory]
    methods: dict[str, dict[str, str]]
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScenarioRow:
    """Metadata describing one coverage row in a contribution matrix.

    One row may correspond to:

    - an exact category contribution row;
    - one finite-CF ambiguity scenario;
    - one regional ambiguity scenario.
    """
    row_id: str
    category_id: str
    category_name: str
    method_id: str
    method_name: str
    scenario_type: str
    scenario_label: str


@dataclass(frozen=True)
class WarningRecord:
    """Structured warning emitted during parsing, modelling, or output."""
    code: str
    message: str
    process_id: str = ""
    process_name: str = ""
    flow_id: str = ""
    flow_name: str = ""
    category_id: str = ""
    category_name: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise the warning into plain JSON/CSV-friendly data."""
        return asdict(self)


@dataclass(frozen=True)
class ProgressEvent:
    """Small progress payload emitted through optional workflow callbacks."""
    step: str
    message: str
    current: int
    total: int


@dataclass
class ScenarioStats:
    """Counts that describe how scenario-row construction behaved.

    These counts help audits answer questions such as:

    - how many rows were exact vs ambiguous?
    - how much regional expansion occurred?
    """
    category_rows: int = 0
    exact_rows: int = 0
    finite_rows: int = 0
    regional_rows: int = 0
    candidate_sets_collapsed: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialise scenario statistics into plain dictionaries."""
        return asdict(self)


@dataclass
class ContributionBundle:
    """The fully characterised process matrix shared by all workflows.

    ``matrix`` has shape ``(n_rows, n_candidate_exchanges)``.  The row metadata
    explains what each row means, while the candidate index list maps columns
    back to exchange positions in the original process payload.
    """
    matrix: Any
    candidate_indices: list[int]
    exchange_keys: list[str]
    flow_ids: list[str]
    characterised: list[bool]
    protected: list[bool]
    row_metadata: list[ScenarioRow]
    stats: ScenarioStats


@dataclass
class ReductionResult:
    """Structured return object for the reduction workflow.

    The object mirrors the sidecars written to disk so notebooks can inspect the
    same information programmatically.
    """
    output_zip: str
    output_dir: str
    metadata_json: str
    warnings_csv: str
    process_manifest_csv: str
    exchange_manifest_csv: str
    validation_json: str
    counts: dict[str, Any]
    warnings: list[WarningRecord]
    diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Serialise the reduction result for notebooks or JSON export."""
        return {
            "output_zip": self.output_zip,
            "output_dir": self.output_dir,
            "metadata_json": self.metadata_json,
            "warnings_csv": self.warnings_csv,
            "process_manifest_csv": self.process_manifest_csv,
            "exchange_manifest_csv": self.exchange_manifest_csv,
            "validation_json": self.validation_json,
            "counts": dict(self.counts),
            "warnings": [item.to_dict() for item in self.warnings],
            "diagnostics": dict(self.diagnostics),
        }


@dataclass
class PriorityResult:
    """Structured return object for the priority-file workflow.

    As with ``ReductionResult``, this keeps the written file paths and the
    in-memory diagnostics aligned.
    """
    output_dir: str
    priority_csv: str
    metadata_json: str
    warnings_csv: str
    counts: dict[str, Any]
    warnings: list[WarningRecord]
    diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Serialise the priority workflow result."""
        return {
            "output_dir": self.output_dir,
            "priority_csv": self.priority_csv,
            "metadata_json": self.metadata_json,
            "warnings_csv": self.warnings_csv,
            "counts": dict(self.counts),
            "warnings": [item.to_dict() for item in self.warnings],
            "diagnostics": dict(self.diagnostics),
        }


@dataclass
class PriorityTauPair:
    """Mapping between one tau value and its compact CSV columns.

    The analyser detects these pairs dynamically from the compact CSV header.
    """
    tau: float
    token: str
    eta_column: str
    loss_column: str
    witness_column: str


@dataclass
class PriorityRow:
    """One parsed row from the compact priority CSV schema.

    The ``metrics`` mapping stores per-tau triples:

    ``(eta, loss_max, witness_string)``
    """
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
    metrics: dict[str, tuple[float, float, str]]


@dataclass
class PriorityDataset:
    """Parsed compact priority data plus optional metadata sidecar.

    This is the analyser's internal loaded representation of the compact output
    files.
    """
    csv_path: str
    rows: list[PriorityRow]
    tau_pairs: list[PriorityTauPair]
    metadata_path: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FlowAnalysisResult:
    """Notebook-friendly result object for compact priority-file analysis.

    The result is deliberately plain: dictionaries and lists rather than a GUI
    view model.  That makes it easy to inspect in notebooks, serialize to JSON,
    or assert against in tests.

    ``loss_max_flows`` is the preferred name for the per-flow detailed section.
    ``selected_flows`` is kept as a backward-compatible alias of the same data.
    """
    requested_flow_ids: list[str]
    requested_flow_names: list[str]
    requested_result_types: list[str]
    tau: float
    matched_flow_ids: list[str]
    unmatched_flow_ids: list[str]
    unmatched_flow_names: list[str]
    ambiguous_names: dict[str, list[str]]
    loss_max_flows: list[dict[str, Any]] | None = None
    selected_flows: list[dict[str, Any]] | None = None
    eta_bounds: dict[str, Any] | None = None
    witness_processes: list[dict[str, Any]] | None = None
    top_n_repair: list[dict[str, Any]] | None = None
    least_n_repair: list[dict[str, Any]] | None = None
    coverage_summary: dict[str, Any] | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialise the analysis report into plain nested Python data."""
        return asdict(self)


def ensure_directory(path: str | Path) -> Path:
    """Create a directory tree if needed and return the resolved ``Path``.

    Centralising the helper keeps output-directory behaviour consistent across
    workflows.
    """
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target
