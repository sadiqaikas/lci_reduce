from __future__ import annotations

import csv
import json

import pytest

from lci_reduce.cf_resolution import (
    CFPromptResult,
    CFResolutionManager,
    CFResolutionSummary,
    load_cf_resolution_choices,
)
from lci_reduce.errors import (
    AmbiguousCharacterisationFactorError,
    DuplicateMethodConflictError,
    UnitCompatibilityError,
)
from lci_reduce.lcia import (
    build_cf_candidate_index,
    check_unit_compatibility,
    collect_categories,
    resolve_cf_for_exchange,
    select_lcia_categories,
)
from lci_reduce.models import (
    CFAmbiguityRecord,
    CharacterisationFactorCandidate,
    FlowInfo,
    ImpactCategory,
    JsonLdArchive,
    UnitInfo,
    WarningRecord,
)
from lci_reduce.validation import build_validation_report


def make_candidate(
    *,
    category_id: str = "cat-1",
    category_name: str = "Human toxicity, non-cancer",
    method_id: str = "method-1",
    method_name: str = "Regional Method",
    flow_id: str = "flow-1",
    flow_name: str = "Emission",
    cf_value: float = 1.0,
    cf_unit: str | None = "kg",
    cf_unit_id: str | None = "unit-kg",
    cf_flow_property_id: str | None = None,
    cf_location_id: str | None = None,
    cf_location_name: str | None = None,
    cf_region: str | None = None,
    cf_compartment: str | None = None,
    cf_subcompartment: str | None = None,
    source_file: str = "category.json",
) -> CharacterisationFactorCandidate:
    return CharacterisationFactorCandidate(
        category_id=category_id,
        category_name=category_name,
        method_id=method_id,
        method_name=method_name,
        flow_id=flow_id,
        flow_name=flow_name,
        cf_value=cf_value,
        cf_unit=cf_unit,
        cf_unit_id=cf_unit_id,
        cf_flow_property_id=cf_flow_property_id,
        cf_flow_property_name=None,
        cf_location_id=cf_location_id,
        cf_location_name=cf_location_name,
        cf_region=cf_region,
        cf_compartment=cf_compartment,
        cf_subcompartment=cf_subcompartment,
        source_file=source_file,
        raw_factor_object={"value": cf_value},
    )


def make_category(
    candidates: list[CharacterisationFactorCandidate],
    *,
    category_id: str = "cat-1",
    category_name: str = "Human toxicity, non-cancer",
    method_id: str = "method-1",
    method_name: str = "Regional Method",
) -> ImpactCategory:
    return ImpactCategory(
        category_id=category_id,
        name=category_name,
        method_id=method_id,
        method_name=method_name,
        path="",
        metadata_text=f"{method_name} {category_name}",
        reference_unit="kg",
        factors={},
        raw={},
        factor_candidates={candidates[0].flow_id: candidates},
        source_file="category.json",
    )


def make_exchange(
    unit_name: str = "kg",
    unit_id: str | None = None,
    location: dict | None = None,
    flow_property: dict | None = None,
) -> dict:
    exchange = {
        "@id": "exchange-1",
        "amount": 1.0,
        "flow": {"@id": "flow-1", "name": "Emission"},
        "unit": {"@id": unit_id or f"unit-{unit_name}", "name": unit_name},
    }
    if location is not None:
        exchange["location"] = location
    if flow_property is not None:
        exchange["flowProperty"] = flow_property
    return exchange


def make_flow(category_path: str = "air/urban air", **kwargs) -> FlowInfo:
    return FlowInfo(
        flow_id="flow-1",
        name="Emission",
        flow_type="ELEMENTARY_FLOW",
        category_path=category_path,
        is_elementary=True,
        raw={},
        location_id=kwargs.get("location_id"),
        location_name=kwargs.get("location_name"),
        location_region=kwargs.get("location_region"),
        reference_flow_property_id=kwargs.get("reference_flow_property_id"),
        reference_flow_property_name=kwargs.get("reference_flow_property_name"),
    )


def make_unit_registry(
    *,
    include_g: bool = True,
    include_kg: bool = True,
    include_mass_flow_property: bool = True,
) -> dict[str, UnitInfo]:
    flow_property_id = "fp-mass" if include_mass_flow_property else None
    flow_property_name = "Mass" if include_mass_flow_property else None
    units: dict[str, UnitInfo] = {}
    if include_kg:
        units["unit-kg"] = UnitInfo(
            unit_id="unit-kg",
            name="kg",
            group_id="group-mass",
            raw={"conversionFactor": 1.0},
            group_name="Units of mass",
            conversion_factor=1.0,
            is_reference_unit=True,
            flow_property_id=flow_property_id,
            flow_property_name=flow_property_name,
        )
    if include_g:
        units["unit-g"] = UnitInfo(
            unit_id="unit-g",
            name="g",
            group_id="group-mass",
            raw={"conversionFactor": 0.001},
            group_name="Units of mass",
            conversion_factor=0.001,
            is_reference_unit=False,
            flow_property_id=flow_property_id,
            flow_property_name=flow_property_name,
        )
    units["unit-m3"] = UnitInfo(
        unit_id="unit-m3",
        name="m3",
        group_id="group-volume",
        raw={"conversionFactor": 1.0},
        group_name="Units of volume",
        conversion_factor=1.0,
        is_reference_unit=True,
        flow_property_id="fp-volume",
        flow_property_name="Volume",
    )
    return units


def test_exact_duplicate_factors():
    candidate_a = make_candidate()
    candidate_b = make_candidate()
    category = make_category([candidate_a, candidate_b])
    resolved = resolve_cf_for_exchange(
        category=category,
        exchange=make_exchange(),
        flow=make_flow(),
        candidates=[candidate_a, candidate_b],
        unit_registry={},
        diagnostic_file="cf_ambiguities.csv",
    )
    assert resolved is not None
    assert resolved.candidate.cf_value == 1.0


def test_exact_duplicate_factors_count_as_automatic_resolution(tmp_path):
    candidate_a = make_candidate()
    candidate_b = make_candidate()
    category = make_category([candidate_a, candidate_b])
    warnings: list[WarningRecord] = []
    manager = CFResolutionManager(mode="cli", choices_path=str(tmp_path / "cf_resolution_choices.csv"))
    resolved = resolve_cf_for_exchange(
        category=category,
        exchange=make_exchange(),
        flow=make_flow(),
        candidates=[candidate_a, candidate_b],
        unit_registry={},
        process_data={"@id": "process-1", "name": "Process"},
        warning_records=warnings,
        diagnostic_file="cf_ambiguities.csv",
        resolution_manager=manager,
    )
    assert resolved is not None
    assert manager.summary.n_cf_ambiguities_found == 1
    assert manager.summary.n_cf_ambiguities_resolved_automatically == 1
    assert any("Collapsed exact duplicate CF candidates" in warning.message for warning in warnings)


def test_category_ref_unit_is_used_as_factor_unit_fallback():
    candidates = build_cf_candidate_index(
        category_data={
            "@id": "category-1",
            "@type": "ImpactCategory",
            "name": "Climate change",
            "impactMethod": {"@id": "method-1", "name": "IPCC"},
            "refUnit": "kg",
            "impactFactors": [
                {
                    "flow": {"@id": "flow-1", "name": "Emission"},
                    "value": 1.0,
                }
            ],
        },
        method_lookup={"method-1": {"name": "IPCC"}},
        path="category.json",
    )
    assert candidates["flow-1"][0].cf_unit == "kg"


def test_conflicting_factors_no_distinguishing_metadata():
    candidate_a = make_candidate(cf_value=1.0)
    candidate_b = make_candidate(cf_value=2.0)
    category = make_category([candidate_a, candidate_b])
    with pytest.raises(AmbiguousCharacterisationFactorError):
        resolve_cf_for_exchange(
            category=category,
            exchange=make_exchange(),
            flow=make_flow(),
            candidates=[candidate_a, candidate_b],
            unit_registry={},
            diagnostic_file="cf_ambiguities.csv",
        )


def test_compartment_disambiguation():
    air_candidate = make_candidate(cf_compartment="air", cf_subcompartment="urban air")
    water_candidate = make_candidate(cf_compartment="water", cf_subcompartment="fresh water")
    category = make_category([air_candidate, water_candidate])
    resolved = resolve_cf_for_exchange(
        category=category,
        exchange=make_exchange(),
        flow=make_flow("air/urban air"),
        candidates=[air_candidate, water_candidate],
        unit_registry={},
        diagnostic_file="cf_ambiguities.csv",
    )
    assert resolved is not None
    assert resolved.candidate.cf_compartment == "air"


def test_missing_flow_compartment_with_compartment_specific_cfs():
    air_candidate = make_candidate(cf_compartment="air")
    water_candidate = make_candidate(cf_compartment="water")
    category = make_category([air_candidate, water_candidate])
    with pytest.raises(AmbiguousCharacterisationFactorError):
        resolve_cf_for_exchange(
            category=category,
            exchange=make_exchange(),
            flow=make_flow(""),
            candidates=[air_candidate, water_candidate],
            unit_registry={},
            diagnostic_file="cf_ambiguities.csv",
        )


def test_unit_conflict_strict_mode():
    candidate = make_candidate(cf_unit="kg", cf_unit_id="unit-kg")
    category = make_category([candidate], method_name="Unit Method")
    with pytest.raises(UnitCompatibilityError):
        resolve_cf_for_exchange(
            category=category,
            exchange=make_exchange("m3"),
            flow=make_flow(),
            candidates=[candidate],
            unit_registry={},
            diagnostic_file="cf_ambiguities.csv",
        )


def test_directly_compatible_unit():
    candidate_kg = make_candidate(cf_unit="kg", cf_unit_id="unit-kg")
    candidate_m3 = make_candidate(cf_unit="m3", cf_unit_id="unit-m3")
    category = make_category([candidate_kg, candidate_m3], method_name="Unit Method")
    resolved = resolve_cf_for_exchange(
        category=category,
        exchange=make_exchange("kg"),
        flow=make_flow(),
        candidates=[candidate_kg, candidate_m3],
        unit_registry={},
        diagnostic_file="cf_ambiguities.csv",
    )
    assert resolved is not None
    assert resolved.candidate.cf_unit == "kg"


def test_single_cf_candidate_same_unit():
    candidate = make_candidate(cf_unit="kg", cf_unit_id="unit-kg", cf_flow_property_id="fp-mass")
    category = make_category([candidate], method_name="Unit Method")
    resolved = resolve_cf_for_exchange(
        category=category,
        exchange=make_exchange("kg", "unit-kg", flow_property={"@id": "fp-mass", "name": "Mass"}),
        flow=make_flow(reference_flow_property_id="fp-mass", reference_flow_property_name="Mass"),
        candidates=[candidate],
        unit_registry=make_unit_registry(),
        diagnostic_file="cf_ambiguities.csv",
    )
    assert resolved is not None
    assert resolved.conversion_factor == 1.0


def test_single_cf_candidate_mass_unit_conversion_kg_to_g():
    candidate = make_candidate(cf_unit="g", cf_unit_id="unit-g", cf_flow_property_id="fp-mass")
    category = make_category([candidate], method_name="Unit Method")
    resolved = resolve_cf_for_exchange(
        category=category,
        exchange=make_exchange("kg", "unit-kg"),
        flow=make_flow(reference_flow_property_id="fp-mass", reference_flow_property_name="Mass"),
        candidates=[candidate],
        unit_registry=make_unit_registry(),
        diagnostic_file="cf_ambiguities.csv",
    )
    assert resolved is not None
    assert resolved.conversion_factor == pytest.approx(1000.0)


def test_single_cf_candidate_mass_unit_conversion_g_to_kg():
    candidate = make_candidate(cf_unit="kg", cf_unit_id="unit-kg", cf_flow_property_id="fp-mass")
    category = make_category([candidate], method_name="Unit Method")
    resolved = resolve_cf_for_exchange(
        category=category,
        exchange=make_exchange("g", "unit-g"),
        flow=make_flow(reference_flow_property_id="fp-mass", reference_flow_property_name="Mass"),
        candidates=[candidate],
        unit_registry=make_unit_registry(),
        diagnostic_file="cf_ambiguities.csv",
    )
    assert resolved is not None
    assert resolved.conversion_factor == pytest.approx(0.001)


def test_exchange_missing_flow_property_uses_flow_reference_property():
    candidate = make_candidate(cf_unit="kg", cf_unit_id="unit-kg", cf_flow_property_id="fp-mass")
    category = make_category([candidate], method_name="Unit Method")
    resolved = resolve_cf_for_exchange(
        category=category,
        exchange=make_exchange("kg", "unit-kg"),
        flow=make_flow(reference_flow_property_id="fp-mass", reference_flow_property_name="Mass"),
        candidates=[candidate],
        unit_registry=make_unit_registry(),
        diagnostic_file="cf_ambiguities.csv",
    )
    assert resolved is not None
    assert resolved.unit_compatibility is not None
    assert resolved.unit_compatibility.flow_property_id == "fp-mass"


def test_single_cf_candidate_incompatible_flow_property():
    candidate = make_candidate(cf_unit="kg", cf_unit_id="unit-kg", cf_flow_property_id="fp-mass")
    category = make_category([candidate], method_name="Unit Method")
    with pytest.raises(UnitCompatibilityError):
        resolve_cf_for_exchange(
            category=category,
            exchange=make_exchange("kg", "unit-kg", flow_property={"@id": "fp-volume", "name": "Volume"}),
            flow=make_flow(reference_flow_property_id="fp-volume", reference_flow_property_name="Volume"),
            candidates=[candidate],
            unit_registry=make_unit_registry(),
            diagnostic_file="cf_ambiguities.csv",
        )


def test_single_cf_candidate_missing_unit_conversion_metadata():
    candidate = make_candidate(cf_unit="g", cf_unit_id="unit-g", cf_flow_property_id="fp-mass")
    category = make_category([candidate], method_name="Unit Method")
    broken_registry = make_unit_registry()
    broken_registry["unit-g"] = UnitInfo(
        unit_id="unit-g",
        name="g",
        group_id="group-mass",
        raw={},
        group_name="Units of mass",
        conversion_factor=None,
        is_reference_unit=False,
        flow_property_id="fp-mass",
        flow_property_name="Mass",
    )
    with pytest.raises(UnitCompatibilityError):
        resolve_cf_for_exchange(
            category=category,
            exchange=make_exchange("kg", "unit-kg"),
            flow=make_flow(reference_flow_property_id="fp-mass", reference_flow_property_name="Mass"),
            candidates=[candidate],
            unit_registry=broken_registry,
            diagnostic_file="cf_ambiguities.csv",
        )


def test_regional_ambiguity():
    eu_candidate = make_candidate(cf_location_id="EU", cf_location_name="Europe", cf_region="EU")
    us_candidate = make_candidate(cf_location_id="US", cf_location_name="United States", cf_region="US")
    category = make_category([eu_candidate, us_candidate], method_name="Regional Method")
    with pytest.raises(AmbiguousCharacterisationFactorError):
        resolve_cf_for_exchange(
            category=category,
            exchange=make_exchange(),
            flow=make_flow(),
            candidates=[eu_candidate, us_candidate],
            unit_registry={},
            process_data={"@id": "process-1", "name": "Process"},
            diagnostic_file="cf_ambiguities.csv",
        )


def test_regional_exact_match():
    eu_candidate = make_candidate(cf_location_id="EU", cf_location_name="Europe", cf_region="EU")
    us_candidate = make_candidate(cf_location_id="US", cf_location_name="United States", cf_region="US")
    category = make_category([eu_candidate, us_candidate], method_name="Regional Method")
    resolved = resolve_cf_for_exchange(
        category=category,
        exchange=make_exchange(location={"@id": "US", "name": "United States", "code": "US"}),
        flow=make_flow(),
        candidates=[eu_candidate, us_candidate],
        unit_registry={},
        diagnostic_file="cf_ambiguities.csv",
    )
    assert resolved is not None
    assert resolved.candidate.cf_location_id == "US"


def test_duplicate_method_uuid_identical_factors():
    candidate = make_candidate()
    category_a = make_category([candidate], category_id="cat-dup")
    category_b = make_category([candidate], category_id="cat-dup")
    archive_a = JsonLdArchive("a", "a", True, [], {}, {}, {}, {"cat-dup": category_a}, {}, [])
    archive_b = JsonLdArchive("b", "b", True, [], {}, {}, {}, {"cat-dup": category_b}, {}, [])
    categories = collect_categories([archive_a, archive_b], diagnostic_file="cf_ambiguities.csv")
    assert len(categories) == 1


def test_duplicate_method_uuid_conflicting_factors():
    category_a = make_category([make_candidate(cf_value=1.0)], category_id="cat-dup")
    category_b = make_category([make_candidate(cf_value=2.0)], category_id="cat-dup")
    archive_a = JsonLdArchive("a", "a", True, [], {}, {}, {}, {"cat-dup": category_a}, {}, [])
    archive_b = JsonLdArchive("b", "b", True, [], {}, {}, {}, {"cat-dup": category_b}, {}, [])
    with pytest.raises(DuplicateMethodConflictError):
        collect_categories([archive_a, archive_b], diagnostic_file="cf_ambiguities.csv")


def test_method_text_selection_matches_multiple_methods():
    category_a = make_category([make_candidate(method_id="m1", method_name="EF v1")], method_id="m1", method_name="EF v1")
    category_b = make_category([make_candidate(method_id="m2", method_name="EF v2")], category_id="cat-2", method_id="m2", method_name="EF v2")
    with pytest.raises(Exception):
        select_lcia_categories([category_a, category_b], "method:EF")


def test_category_text_selection_matches_multiple_categories():
    category_a = make_category([make_candidate()], category_id="cat-1", category_name="Human toxicity")
    category_b = make_category([make_candidate(category_id="cat-2", category_name="Human toxicity, non-cancer")], category_id="cat-2", category_name="Human toxicity, non-cancer")
    with pytest.raises(Exception):
        select_lcia_categories([category_a, category_b], "category:Human toxicity")


def test_validation_counter_for_unit_failure():
    warning = WarningRecord(
        severity="error",
        object_type="characterisation_factor",
        object_id="cat-1",
        object_name="Global warming",
        process_id="process-1",
        process_name="Coal plant",
        flow_id="flow-1",
        flow_name="Carbon dioxide, fossil",
        category_id="cat-1",
        category_name="Global warming",
        message="exchange_flow_property_id=fp-mass; reason=missing_explicit_unit_conversion",
    )
    ambiguity = CFAmbiguityRecord(
        severity="error",
        method_id="method-1",
        method_name="IPCC",
        category_id="cat-1",
        category_name="Global warming",
        flow_id="flow-1",
        flow_name="Carbon dioxide, fossil",
        candidate_count=1,
        candidate_index=0,
        cf_value="1",
        cf_unit="g",
        cf_unit_id="unit-g",
        cf_flow_property_id="fp-mass",
        cf_flow_property_name="Mass",
        cf_compartment="air",
        cf_subcompartment="urban air",
        cf_location_id="",
        cf_location_name="",
        cf_region="",
        exchange_unit="kg",
        exchange_unit_id="unit-kg",
        exchange_flow_property_id="",
        exchange_flow_property_name="",
        flow_reference_flow_property_id="fp-mass",
        flow_reference_flow_property_name="Mass",
        source_file="category.json",
        differing_fields="cf_unit,cf_flow_property_id",
        message="unit conflict",
        issue_type="unit_conflict",
        group_key="unit:cat-1:flow-1:process-1",
    )
    report = build_validation_report(
        reduced_results=[],
        selected_categories=[],
        empty_selected_categories=[],
        warnings=[warning],
        cf_ambiguities=[ambiguity],
        output_zip="",
        pdf_report="",
        cf_ambiguities_csv="cf_ambiguities.csv",
        cf_resolution_choices_csv="cf_resolution_choices.csv",
        cf_resolution_summary=CFResolutionSummary(
            n_cf_ambiguities_found=3,
            n_cf_ambiguity_keys_unique=2,
            n_cf_ambiguities_resolved_automatically=1,
            n_cf_unique_user_decisions=1,
            n_cf_ambiguities_resolved_by_user_choice=1,
            n_cf_ambiguities_unresolved=1,
            n_cf_resolution_choices_reused=1,
        ),
    )
    assert report["n_unit_failures"] == 1
    assert report["n_cf_unit_conflicts"] == 1
    assert report["n_missing_flow_failures"] == 0
    assert report["n_cf_ambiguities_found"] == 3
    assert report["n_cf_ambiguity_keys_unique"] == 2
    assert report["n_cf_unique_user_decisions"] == 1
    assert report["n_cf_ambiguities_resolved_by_user_choice"] == 1
    assert report["cf_resolution_choices_csv"] == "cf_resolution_choices.csv"


def test_user_choice_is_written_to_cf_resolution_choices_csv(tmp_path):
    candidate_a = make_candidate(cf_value=1.0, source_file="category-a.json")
    candidate_b = make_candidate(cf_value=2.0, source_file="category-b.json")
    category = make_category([candidate_a, candidate_b])
    choices_path = tmp_path / "cf_resolution_choices.csv"
    manager = CFResolutionManager(
        mode="gui",
        choices_path=str(choices_path),
        prompt=lambda context, candidates: CFPromptResult(action="select", candidate_index=1),
    )

    resolved = resolve_cf_for_exchange(
        category=category,
        exchange=make_exchange(),
        flow=make_flow(),
        candidates=[candidate_a, candidate_b],
        unit_registry={},
        process_data={"@id": "process-1", "name": "Process"},
        diagnostic_file="cf_ambiguities.csv",
        resolution_manager=manager,
    )

    assert resolved is not None
    assert resolved.candidate.cf_value == 2.0
    assert choices_path.exists()
    loaded = load_cf_resolution_choices(choices_path)
    record = loaded[("method-1", "cat-1", "flow-1")]
    assert record.chosen_cf_value == "2"
    assert record.process_id == "process-1"
    assert record.exchange_id == "exchange-1"
    assert record.exchange_index == "-1"
    assert record.choice_origin == "new"
    assert json.loads(record.rejected_cf_values) == ["1"]
    rejected_metadata = json.loads(record.rejected_candidate_metadata)
    assert rejected_metadata[0]["source_file"] == "category-a.json"
    assert "category-b.json" in record.chosen_candidate_metadata
    raw_rows = list(csv.DictReader(choices_path.open("r", encoding="utf-8", newline="")))
    assert len(raw_rows) == 1
    assert json.loads(raw_rows[0]["all_candidate_cf_values"]) == ["1", "2"]
    assert manager.summary.n_cf_ambiguities_resolved_by_user_choice == 1
    assert manager.summary.n_cf_unique_user_decisions == 1


def test_saved_choice_is_reused(tmp_path):
    candidate_a = make_candidate(cf_value=1.0, source_file="category-a.json")
    candidate_b = make_candidate(cf_value=2.0, source_file="category-b.json")
    category = make_category([candidate_a, candidate_b])
    choices_path = tmp_path / "cf_resolution_choices.csv"

    first_manager = CFResolutionManager(
        mode="gui",
        choices_path=str(choices_path),
        prompt=lambda context, candidates: CFPromptResult(action="select", candidate_index=1),
    )
    resolve_cf_for_exchange(
        category=category,
        exchange=make_exchange(),
        flow=make_flow(),
        candidates=[candidate_a, candidate_b],
        unit_registry={},
        process_data={"@id": "process-1", "name": "Process"},
        diagnostic_file="cf_ambiguities.csv",
        resolution_manager=first_manager,
    )

    warnings: list[WarningRecord] = []
    second_manager = CFResolutionManager(mode="cli", choices_path=str(choices_path))
    resolved = resolve_cf_for_exchange(
        category=category,
        exchange=make_exchange(),
        flow=make_flow(),
        candidates=[candidate_a, candidate_b],
        unit_registry={},
        process_data={"@id": "process-2", "name": "Process 2"},
        warning_records=warnings,
        diagnostic_file="cf_ambiguities.csv",
        resolution_manager=second_manager,
    )

    assert resolved is not None
    assert resolved.candidate.cf_value == 2.0
    assert second_manager.summary.n_cf_ambiguities_resolved_by_user_choice == 1
    assert second_manager.summary.n_cf_resolution_choices_reused == 1
    assert second_manager.summary.n_cf_unique_user_decisions == 0
    assert any("resolved reused_choice" in warning.message for warning in warnings)

    rows = list(csv.DictReader(choices_path.open("r", encoding="utf-8", newline="")))
    assert len(rows) == 2
    assert rows[0]["choice_origin"] == "new"
    assert rows[1]["choice_origin"] == "reused"
    assert json.loads(rows[1]["rejected_cf_values"]) == ["1"]


def test_saved_choice_is_not_reused_when_candidate_set_changes(tmp_path):
    candidate_a = make_candidate(cf_value=1.0, source_file="category-a.json")
    candidate_b = make_candidate(cf_value=2.0, source_file="category-b.json")
    category = make_category([candidate_a, candidate_b])
    choices_path = tmp_path / "cf_resolution_choices.csv"

    first_manager = CFResolutionManager(
        mode="gui",
        choices_path=str(choices_path),
        prompt=lambda context, candidates: CFPromptResult(action="select", candidate_index=1),
    )
    resolve_cf_for_exchange(
        category=category,
        exchange=make_exchange(),
        flow=make_flow(),
        candidates=[candidate_a, candidate_b],
        unit_registry={},
        process_data={"@id": "process-1", "name": "Process"},
        diagnostic_file="cf_ambiguities.csv",
        resolution_manager=first_manager,
    )

    changed_candidate_b = make_candidate(cf_value=2.0, source_file="category-b-reissued.json")
    second_manager = CFResolutionManager(mode="cli", choices_path=str(choices_path))
    with pytest.raises(AmbiguousCharacterisationFactorError):
        resolve_cf_for_exchange(
            category=category,
            exchange=make_exchange(),
            flow=make_flow(),
            candidates=[candidate_a, changed_candidate_b],
            unit_registry={},
            process_data={"@id": "process-2", "name": "Process 2"},
            diagnostic_file="cf_ambiguities.csv",
            resolution_manager=second_manager,
        )

    assert second_manager.summary.n_cf_resolution_choices_reused == 0
    rows = list(csv.DictReader(choices_path.open("r", encoding="utf-8", newline="")))
    assert [row["choice_origin"] for row in rows] == ["new"]


def test_unresolved_ambiguity_stops_the_run(tmp_path):
    candidate_a = make_candidate(cf_value=1.0)
    candidate_b = make_candidate(cf_value=2.0)
    category = make_category([candidate_a, candidate_b])
    manager = CFResolutionManager(
        mode="gui",
        choices_path=str(tmp_path / "cf_resolution_choices.csv"),
        prompt=lambda context, candidates: CFPromptResult(action="skip_fail"),
    )

    with pytest.raises(AmbiguousCharacterisationFactorError):
        resolve_cf_for_exchange(
            category=category,
            exchange=make_exchange(),
            flow=make_flow(),
            candidates=[candidate_a, candidate_b],
            unit_registry={},
            process_data={"@id": "process-1", "name": "Process"},
            diagnostic_file="cf_ambiguities.csv",
            resolution_manager=manager,
        )

    assert manager.summary.n_cf_ambiguities_found == 1
    assert manager.summary.n_cf_ambiguities_unresolved == 1
