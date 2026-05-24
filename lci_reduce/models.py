"""Shared data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class DatasetEntry:
    path: str
    raw_bytes: bytes
    data: Dict[str, Any]
    object_type: str
    object_id: str
    name: str


@dataclass
class DatasetLocator:
    path: str
    object_type: str
    object_id: str
    name: str


@dataclass
class FlowInfo:
    flow_id: str
    name: str
    flow_type: str
    category_path: str
    is_elementary: bool
    raw: Dict[str, Any]
    location_id: Optional[str] = None
    location_name: Optional[str] = None
    location_region: Optional[str] = None
    reference_flow_property_id: Optional[str] = None
    reference_flow_property_name: Optional[str] = None


@dataclass
class UnitInfo:
    unit_id: str
    name: str
    group_id: Optional[str]
    raw: Dict[str, Any]
    group_name: Optional[str] = None
    conversion_factor: Optional[float] = None
    is_reference_unit: bool = False
    flow_property_id: Optional[str] = None
    flow_property_name: Optional[str] = None


@dataclass
class CharacterizationFactor:
    flow_id: str
    value: float
    unit_name: Optional[str]
    raw: Dict[str, Any]


@dataclass
class CharacterisationFactorCandidate:
    category_id: str
    category_name: str
    method_id: Optional[str]
    method_name: Optional[str]
    flow_id: str
    flow_name: str
    cf_value: float
    cf_unit: Optional[str]
    cf_unit_id: Optional[str]
    cf_flow_property_id: Optional[str]
    cf_flow_property_name: Optional[str]
    cf_location_id: Optional[str]
    cf_location_name: Optional[str]
    cf_region: Optional[str]
    cf_compartment: Optional[str]
    cf_subcompartment: Optional[str]
    source_file: str
    raw_factor_object: Dict[str, Any]


@dataclass
class ResolvedCharacterisationFactor:
    candidate: CharacterisationFactorCandidate
    candidate_count: int
    differing_fields: List[str]
    conversion_factor: float = 1.0
    unit_compatibility: Optional["UnitCompatibilityResult"] = None


@dataclass
class CFAmbiguityRecord:
    severity: str
    method_id: str
    method_name: str
    category_id: str
    category_name: str
    flow_id: str
    flow_name: str
    candidate_count: int
    candidate_index: int
    cf_value: str
    cf_unit: str
    cf_unit_id: str
    cf_flow_property_id: str
    cf_flow_property_name: str
    cf_compartment: str
    cf_subcompartment: str
    cf_location_id: str
    cf_location_name: str
    cf_region: str
    exchange_unit: str
    exchange_unit_id: str
    exchange_flow_property_id: str
    exchange_flow_property_name: str
    flow_reference_flow_property_id: str
    flow_reference_flow_property_name: str
    source_file: str
    differing_fields: str
    message: str
    issue_type: str = ""
    group_key: str = ""
    process_id: str = ""
    process_name: str = ""
    exchange_id: str = ""
    exchange_index: str = ""
    ambiguity_key: str = ""
    resolution_status: str = ""
    choice_origin: str = ""
    occurrence_timestamp: str = ""
    all_candidate_cf_values: str = ""
    all_candidate_metadata: str = ""
    chosen_cf_value: str = ""
    rejected_cf_values: str = ""
    candidate_selected: str = ""


@dataclass
class UnitCompatibilityResult:
    compatible: bool
    conversion_factor: float
    reason: str
    exchange_unit_id: Optional[str]
    exchange_unit_name: Optional[str]
    exchange_flow_property_id: Optional[str]
    exchange_flow_property_name: Optional[str]
    flow_reference_flow_property_id: Optional[str]
    flow_reference_flow_property_name: Optional[str]
    cf_unit_id: Optional[str]
    cf_unit_name: Optional[str]
    cf_flow_property_id: Optional[str]
    cf_flow_property_name: Optional[str]
    flow_property_id: Optional[str]
    flow_property_name: Optional[str]


@dataclass
class ImpactCategory:
    category_id: str
    name: str
    method_id: Optional[str]
    method_name: Optional[str]
    path: str
    metadata_text: str
    reference_unit: Optional[str]
    factors: Dict[str, CharacterizationFactor]
    raw: Dict[str, Any]
    factor_candidates: Dict[str, List[CharacterisationFactorCandidate]] = field(default_factory=dict)
    source_file: str = ""
    method_path: str = ""
    method_source_file: str = ""


@dataclass
class ImpactCategoryReport:
    category_id: str
    category_name: str
    method_id: str
    method_name: str
    method_path: str
    source_file: str


@dataclass
class JsonLdArchive:
    source_path: str
    source_name: str
    is_zip: bool
    entries: List[DatasetEntry]
    processes: Dict[str, DatasetEntry]
    flows: Dict[str, FlowInfo]
    units: Dict[str, UnitInfo]
    impact_categories: Dict[str, ImpactCategory]
    impact_methods: Dict[str, DatasetEntry]
    other_entries: List[DatasetEntry]


@dataclass
class JsonLdArchiveIndex:
    source_path: str
    source_name: str
    resolved_source_path: str
    is_zip: bool
    entry_paths: List[str]
    processes: Dict[str, DatasetLocator]
    flows: Dict[str, FlowInfo]
    units: Dict[str, UnitInfo]
    impact_categories: Dict[str, ImpactCategory]


@dataclass
class WarningRecord:
    severity: str
    object_type: str
    object_id: str
    object_name: str
    process_id: str
    process_name: str
    flow_id: str
    flow_name: str
    category_id: str
    category_name: str
    message: str
    method_id: str = ""
    method_name: str = ""
    source_file: str = ""


@dataclass
class ProcessReductionResult:
    process_id: str
    process_name: str
    process_path: str
    original_process: Dict[str, Any]
    reduced_process: Dict[str, Any]
    selected_mask: np.ndarray
    selected_pos_mask: np.ndarray
    selected_neg_mask: np.ndarray
    candidate_indices: List[int]
    kept_indices: List[int]
    removed_indices: List[int]
    elementary_rows: List[Dict[str, Any]]
    process_row: Dict[str, Any]
    full_pos: np.ndarray
    full_neg: np.ndarray
    retained_pos: np.ndarray
    retained_neg: np.ndarray
    active_pos: np.ndarray
    active_neg: np.ndarray
    n_uncharacterised_kept: int = 0
    n_uncharacterised_removed: int = 0
    warnings: List[WarningRecord] = field(default_factory=list)


@dataclass
class InspectResult:
    n_processes: int
    n_flows: int
    n_elementary_flows: int
    n_methods: int
    n_categories: int
    database_methods: int
    database_categories: int
    database_contains_impact_methods: bool
    method_names: List[str]
    family_names: List[str]
    cf_mappings_buildable: bool
    categories_with_factors: int
    lcia_categories: List[ImpactCategoryReport] = field(default_factory=list)
    empty_lcia_categories: List[ImpactCategoryReport] = field(default_factory=list)


@dataclass
class CreateConfig:
    database: str
    methods: Optional[str]
    output_dir: str
    tau: float
    method_selection: str
    uncharacterised_policy: str
    strict_units: bool
    tolerance: float
    cf_resolution_file: Optional[str] = None


@dataclass
class CreateResult:
    output_zip: str
    exchange_manifest_csv: str
    process_manifest_csv: str
    run_summary_json: str
    summary: Dict[str, Any]


@dataclass
class FlowPriorityConfig:
    database: str
    methods: Optional[str]
    output_dir: str
    method_selection: str
    audit_tau_values: List[float]
    strict_units: bool
    tolerance: float
    cf_resolution_file: Optional[str] = None


@dataclass
class FlowPriorityResult:
    flow_priority_csv: str
    flow_priority_metadata_json: str
    metadata: Dict[str, Any]


@dataclass
class CreateProgressUpdate:
    step: str
    message: str
    current: int
    total: int
    stage_current: int | None = None
    stage_total: int | None = None
    process_current: int | None = None
    process_total: int | None = None
    process_name: str = ""
    tau: float | None = None
    n_elementary_before: int | None = None
    n_elementary_removed: int | None = None


@dataclass
class TauReductionRun:
    id: str
    sourceFileName: str
    tau: float | None
    elementaryBefore: int | None
    elementaryAfter: int | None
    elementaryRemoved: int | None
    retainedPercent: float | None
    removedPercent: float | None
    validationStatus: str
    warnings: List[str] = field(default_factory=list)
    inputDatabaseName: Optional[str] = None
    inputDatabaseHash: Optional[str] = None
    sourcePath: str = ""
    runDirectory: str = ""
    state: str = "ready"
    statusMessage: str = ""
    sourceWarnings: List[str] = field(default_factory=list)
    groupWarnings: List[str] = field(default_factory=list)

    def refresh_warnings(self) -> None:
        self.warnings = [*self.sourceWarnings, *self.groupWarnings]


@dataclass
class DatabaseReductionGroup:
    id: str
    name: str
    runs: List[TauReductionRun] = field(default_factory=list)


CharacterisationFactor = CharacterizationFactor
CharacterizationFactorCandidate = CharacterisationFactorCandidate
