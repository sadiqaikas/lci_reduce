from __future__ import annotations

import csv
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from lci_reduce.contribution import build_contribution_details
from lci_reduce.errors import ScenarioExpansionError
from lci_reduce.flow_priority import single_flow_shortfall
from lci_reduce.cli import priority_command
from lci_reduce.models import CharacterisationFactorCandidate, FlowInfo, ImpactCategory
from lci_reduce.reducer import signed_tau_cover


def make_candidate(
    *,
    flow_id: str,
    flow_name: str,
    cf_value: float,
    category_id: str = "cat-1",
    category_name: str = "Climate change",
    method_id: str = "method-1",
    method_name: str = "Method",
    cf_location_id: str | None = None,
    cf_location_name: str | None = None,
    cf_region: str | None = None,
    raw_factor_object: dict | None = None,
) -> CharacterisationFactorCandidate:
    return CharacterisationFactorCandidate(
        category_id=category_id,
        category_name=category_name,
        method_id=method_id,
        method_name=method_name,
        flow_id=flow_id,
        flow_name=flow_name,
        cf_value=cf_value,
        cf_unit="kg",
        cf_unit_id=None,
        cf_flow_property_id=None,
        cf_flow_property_name=None,
        cf_location_id=cf_location_id,
        cf_location_name=cf_location_name,
        cf_region=cf_region,
        cf_compartment=None,
        cf_subcompartment=None,
        source_file="category.json",
        raw_factor_object=raw_factor_object or {"value": cf_value},
    )


def make_category(
    factor_candidates: dict[str, list[CharacterisationFactorCandidate]],
    *,
    category_id: str = "cat-1",
    category_name: str = "Climate change",
    method_id: str = "method-1",
    method_name: str = "Method",
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
        factor_candidates=factor_candidates,
        source_file="category.json",
    )


def make_flow(flow_id: str, flow_name: str) -> FlowInfo:
    return FlowInfo(
        flow_id=flow_id,
        name=flow_name,
        flow_type="ELEMENTARY_FLOW",
        category_path="air/urban air",
        is_elementary=True,
        raw={},
    )


def make_exchange(flow_id: str, flow_name: str, amount: float, *, location: dict | None = None) -> dict:
    exchange = {
        "@id": flow_id.replace("flow-", "ex-"),
        "amount": amount,
        "flow": {"@id": flow_id, "name": flow_name},
        "unit": {"name": "kg"},
    }
    if location is not None:
        exchange["location"] = location
    return exchange


def make_location_axis_candidates(
    *,
    flow_id: str,
    flow_name: str,
    location_values: Sequence[tuple[str, float]],
    generic_values: Sequence[float] = (),
) -> list[CharacterisationFactorCandidate]:
    candidates = [
        make_candidate(
            flow_id=flow_id,
            flow_name=flow_name,
            cf_value=value,
            cf_location_id=location,
            cf_location_name=location,
        )
        for location, value in location_values
    ]
    candidates.extend(
        make_candidate(
            flow_id=flow_id,
            flow_name=flow_name,
            cf_value=value,
        )
        for value in generic_values
    )
    return candidates


def make_location_axis_values(count: int, *, start: int = 1) -> list[tuple[str, float]]:
    return [(f"L{index:02d}", float(start + index - 1)) for index in range(1, count + 1)]


def build_single_category_matrix(
    *,
    exchanges: list[dict],
    flows: dict[str, FlowInfo],
    category: ImpactCategory,
    process_data: dict | None = None,
    max_scenario_rows_per_process: int = 300000,
    max_candidate_set_size: int = 500,
):
    return build_contribution_details(
        exchanges=exchanges,
        flow_lookup=flows,
        categories=[category],
        unit_registry={},
        strict_units=True,
        tol=1e-12,
        process_data=process_data,
        max_scenario_rows_per_process=max_scenario_rows_per_process,
        max_candidate_set_size=max_candidate_set_size,
    )


def _write_archive(path: Path, files: dict[str, dict]) -> Path:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for rel_path, data in files.items():
            archive.writestr(rel_path, json.dumps(data))
    return path


def _read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames is not None
        return list(reader.fieldnames), list(reader)


def make_priority_scenario_zip(base: Path, name: str = "priority_scenario_db.zip") -> Path:
    return _write_archive(
        base / name,
        {
            "processes/process-1.json": {
                "@id": "process-1",
                "@type": "Process",
                "name": "Scenario Process",
                "exchanges": [
                    {
                        "@id": "product",
                        "amount": 1.0,
                        "flow": {"@id": "flow-product", "name": "Product"},
                        "unit": {"name": "kg"},
                        "quantitativeReference": True,
                    },
                    {
                        "@id": "ex-exact",
                        "amount": 80.0,
                        "flow": {"@id": "flow-exact", "name": "Exact flow"},
                        "unit": {"name": "kg"},
                    },
                    {
                        "@id": "ex-amb",
                        "amount": 1.0,
                        "flow": {"@id": "flow-amb", "name": "Ambiguous flow"},
                        "unit": {"name": "kg"},
                    },
                ],
            },
            "flows/flow-product.json": {
                "@id": "flow-product",
                "@type": "Flow",
                "name": "Product",
                "flowType": "PRODUCT_FLOW",
                "categoryPath": "products",
            },
            "flows/flow-exact.json": {
                "@id": "flow-exact",
                "@type": "Flow",
                "name": "Exact flow",
                "flowType": "ELEMENTARY_FLOW",
                "categoryPath": "air/urban air",
            },
            "flows/flow-amb.json": {
                "@id": "flow-amb",
                "@type": "Flow",
                "name": "Ambiguous flow",
                "flowType": "ELEMENTARY_FLOW",
                "categoryPath": "air/urban air",
            },
            "lcia_methods/method-1.json": {
                "@id": "method-1",
                "@type": "ImpactMethod",
                "name": "Scenario Method",
            },
            "lcia_categories/category-1.json": {
                "@id": "category-1",
                "@type": "ImpactCategory",
                "name": "Scenario Climate",
                "impactMethod": {"@id": "method-1", "name": "Scenario Method"},
                "referenceUnitName": "kg",
                "impactFactors": [
                    {"flow": {"@id": "flow-exact", "name": "Exact flow"}, "value": 1.0, "unitName": "kg"},
                    {"flow": {"@id": "flow-amb", "name": "Ambiguous flow"}, "value": 20.0, "unitName": "kg"},
                    {"flow": {"@id": "flow-amb", "name": "Ambiguous flow"}, "value": 100.0, "unitName": "kg"},
                ],
            },
        },
    )


def test_exact_cf_behaviour_is_unchanged() -> None:
    flows = {
        "flow-a": make_flow("flow-a", "A"),
        "flow-b": make_flow("flow-b", "B"),
    }
    category = make_category(
        {
            "flow-a": [make_candidate(flow_id="flow-a", flow_name="A", cf_value=2.0)],
            "flow-b": [make_candidate(flow_id="flow-b", flow_name="B", cf_value=3.0)],
        }
    )
    matrix, candidate_indices, _exchange_keys, _characterised, _resolved_mask, row_metadata, stats = build_single_category_matrix(
        exchanges=[
            make_exchange("flow-a", "A", 10.0),
            make_exchange("flow-b", "B", 5.0),
        ],
        flows=flows,
        category=category,
        process_data={"@id": "process-1", "name": "Exact Process"},
    )
    assert candidate_indices == [0, 1]
    assert matrix.shape == (1, 2)
    assert np.allclose(matrix[0], np.array([20.0, 15.0]))
    assert row_metadata[0].scenario_type == "exact"
    assert stats.n_finite_cf_candidate_sets == 0


def test_complete_rows_include_exact_contributions_not_component_rows() -> None:
    flows = {
        "flow-exact": make_flow("flow-exact", "Exact"),
        "flow-amb": make_flow("flow-amb", "Ambiguous"),
    }
    category = make_category(
        {
            "flow-exact": [make_candidate(flow_id="flow-exact", flow_name="Exact", cf_value=1.0)],
            "flow-amb": [
                make_candidate(flow_id="flow-amb", flow_name="Ambiguous", cf_value=20.0),
                make_candidate(flow_id="flow-amb", flow_name="Ambiguous", cf_value=100.0),
            ],
        }
    )
    matrix, _candidate_indices, _exchange_keys, _characterised, _resolved_mask, row_metadata, _stats = build_single_category_matrix(
        exchanges=[
            make_exchange("flow-exact", "Exact", 80.0),
            make_exchange("flow-amb", "Ambiguous", 1.0),
        ],
        flows=flows,
        category=category,
        process_data={"@id": "process-1", "name": "Scenario Process"},
    )
    assert matrix.shape == (2, 2)
    assert row_metadata[0].scenario_type == "finite_cf"
    assert row_metadata[1].scenario_type == "finite_cf"
    assert np.allclose(matrix, np.array([[80.0, 20.0], [80.0, 100.0]]))


def test_greedy_selects_one_retained_set_satisfying_all_scenario_rows() -> None:
    flows = {
        "flow-exact": make_flow("flow-exact", "Exact"),
        "flow-amb": make_flow("flow-amb", "Ambiguous"),
    }
    category = make_category(
        {
            "flow-exact": [make_candidate(flow_id="flow-exact", flow_name="Exact", cf_value=1.0)],
            "flow-amb": [
                make_candidate(flow_id="flow-amb", flow_name="Ambiguous", cf_value=20.0),
                make_candidate(flow_id="flow-amb", flow_name="Ambiguous", cf_value=100.0),
            ],
        }
    )
    matrix, _candidate_indices, exchange_keys, _characterised, _resolved_mask, _row_metadata, _stats = build_single_category_matrix(
        exchanges=[
            make_exchange("flow-exact", "Exact", 80.0),
            make_exchange("flow-amb", "Ambiguous", 1.0),
        ],
        flows=flows,
        category=category,
        process_data={"@id": "process-1", "name": "Scenario Process"},
    )
    result = signed_tau_cover(matrix, 0.95, exchange_keys=exchange_keys)
    assert result["selected"].tolist() == [True, True]
    retained = matrix[:, result["selected"]].sum(axis=1)
    full = matrix.sum(axis=1)
    active = full > 1e-12
    assert np.all(retained[active] >= 0.95 * full[active] - 1e-12)


def test_flow_priority_uses_complete_scenario_rows_for_eta_and_loss_max(tmp_path: Path) -> None:
    db_zip = make_priority_scenario_zip(tmp_path)
    output_dir = tmp_path / "priority_out"
    result = priority_command(
        database=str(db_zip),
        methods=None,
        output=str(output_dir),
        method_selection="all",
        audit_tau=[0.95],
        strict_units=True,
        tolerance=1e-12,
    )
    _header, rows = _read_csv_rows(Path(result.flow_priority_csv))
    row_lookup = {row["flow_id"]: row for row in rows}
    ambiguous = row_lookup["flow-amb"]
    expected_loss = 100.0 / 180.0
    expected_eta = single_flow_shortfall(0.95, 1.0, expected_loss)
    assert float(ambiguous["loss_max_0_95"]) == pytest.approx(expected_loss)
    assert float(ambiguous["eta_0_95"]) == pytest.approx(expected_eta)


def test_priority_witness_includes_scenario_label(tmp_path: Path) -> None:
    db_zip = make_priority_scenario_zip(tmp_path)
    output_dir = tmp_path / "priority_witness_out"
    result = priority_command(
        database=str(db_zip),
        methods=None,
        output=str(output_dir),
        method_selection="all",
        audit_tau=[0.95],
        strict_units=True,
        tolerance=1e-12,
    )
    _header, rows = _read_csv_rows(Path(result.flow_priority_csv))
    row_lookup = {row["flow_id"]: row for row in rows}
    witness = row_lookup["flow-amb"]["eta_0_95_witness"]
    assert "Scenario Climate" in witness
    assert "exchange:ex-amb" in witness
    assert "cf=100" in witness


def test_shared_regional_location_creates_shared_rows_not_cartesian() -> None:
    flows = {
        "flow-a": make_flow("flow-a", "A"),
        "flow-b": make_flow("flow-b", "B"),
    }
    category = make_category(
        {
            "flow-a": [
                make_candidate(flow_id="flow-a", flow_name="A", cf_value=10.0, cf_location_id="GB", cf_location_name="GB"),
                make_candidate(flow_id="flow-a", flow_name="A", cf_value=100.0, cf_location_id="US", cf_location_name="US"),
            ],
            "flow-b": [
                make_candidate(flow_id="flow-b", flow_name="B", cf_value=1.0, cf_location_id="GB", cf_location_name="GB"),
                make_candidate(flow_id="flow-b", flow_name="B", cf_value=50.0, cf_location_id="US", cf_location_name="US"),
            ],
        }
    )
    matrix, _candidate_indices, _exchange_keys, _characterised, _resolved_mask, row_metadata, _stats = build_single_category_matrix(
        exchanges=[
            make_exchange("flow-a", "A", 1.0),
            make_exchange("flow-b", "B", 1.0),
        ],
        flows=flows,
        category=category,
        process_data={"@id": "process-1", "name": "Regional Process", "location": {"name": "EU"}},
    )
    assert matrix.shape == (2, 2)
    assert all(row.scenario_type == "regional_cf" for row in row_metadata)
    assert np.allclose(matrix, np.array([[10.0, 1.0], [100.0, 50.0]]))


def test_identical_37_location_axes_create_37_rows_not_cartesian() -> None:
    flows = {
        "flow-a": make_flow("flow-a", "A"),
        "flow-b": make_flow("flow-b", "B"),
        "flow-c": make_flow("flow-c", "C"),
    }
    axis_values = make_location_axis_values(37, start=10)
    category = make_category(
        {
            "flow-a": make_location_axis_candidates(flow_id="flow-a", flow_name="A", location_values=axis_values),
            "flow-b": make_location_axis_candidates(flow_id="flow-b", flow_name="B", location_values=axis_values),
            "flow-c": make_location_axis_candidates(flow_id="flow-c", flow_name="C", location_values=axis_values),
        }
    )
    matrix, _candidate_indices, _exchange_keys, _characterised, _resolved_mask, row_metadata, stats = build_single_category_matrix(
        exchanges=[
            make_exchange("flow-a", "A", 1.0),
            make_exchange("flow-b", "B", 1.0),
            make_exchange("flow-c", "C", 1.0),
        ],
        flows=flows,
        category=category,
        process_data={"@id": "process-1", "name": "Regional Fallback Process", "location": {"name": "CH"}},
    )
    assert matrix.shape == (37, 3)
    assert all(row.scenario_type == "regional_cf" for row in row_metadata)
    assert stats.n_regional_scenario_groups == 1
    assert stats.n_independent_candidate_groups == 0


def test_identical_37_location_axes_with_generic_fallback_do_not_add_generic_row() -> None:
    flows = {
        "flow-a": make_flow("flow-a", "A"),
        "flow-b": make_flow("flow-b", "B"),
        "flow-c": make_flow("flow-c", "C"),
    }
    axis_values = make_location_axis_values(37, start=10)
    category = make_category(
        {
            "flow-a": make_location_axis_candidates(flow_id="flow-a", flow_name="A", location_values=axis_values, generic_values=[999.0]),
            "flow-b": make_location_axis_candidates(flow_id="flow-b", flow_name="B", location_values=axis_values, generic_values=[999.0]),
            "flow-c": make_location_axis_candidates(flow_id="flow-c", flow_name="C", location_values=axis_values, generic_values=[999.0]),
        }
    )
    matrix, _candidate_indices, _exchange_keys, _characterised, _resolved_mask, row_metadata, _stats = build_single_category_matrix(
        exchanges=[
            make_exchange("flow-a", "A", 1.0),
            make_exchange("flow-b", "B", 1.0),
            make_exchange("flow-c", "C", 1.0),
        ],
        flows=flows,
        category=category,
        process_data={"@id": "process-1", "name": "Axis With Fallback Process", "location": {"name": "CH"}},
    )
    assert matrix.shape == (37, 3)
    assert all(row.scenario_type == "regional_cf" for row in row_metadata)
    assert not any("generic/unlocated" in row.scenario_label for row in row_metadata)


def test_subset_location_axis_with_generic_fallback_stays_shared() -> None:
    flows = {
        "flow-a": make_flow("flow-a", "A"),
        "flow-b": make_flow("flow-b", "B"),
    }
    full_axis = make_location_axis_values(37, start=10)
    subset_axis = full_axis[:30]
    category = make_category(
        {
            "flow-a": make_location_axis_candidates(flow_id="flow-a", flow_name="A", location_values=full_axis),
            "flow-b": make_location_axis_candidates(flow_id="flow-b", flow_name="B", location_values=subset_axis, generic_values=[500.0]),
        }
    )
    matrix, _candidate_indices, _exchange_keys, _characterised, _resolved_mask, row_metadata, stats = build_single_category_matrix(
        exchanges=[
            make_exchange("flow-a", "A", 1.0),
            make_exchange("flow-b", "B", 1.0),
        ],
        flows=flows,
        category=category,
        process_data={"@id": "process-1", "name": "Subset Fallback Process"},
    )
    assert matrix.shape == (37, 2)
    assert stats.n_regional_scenario_groups == 1
    assert stats.n_independent_candidate_groups == 0
    assert any("fallback_for_shared_region=yes" in row.scenario_label for row in row_metadata)
    assert any(tuple(row.tolist()) == (40.0, 500.0) for row in matrix)


def test_fixed_single_option_contribution_is_copied_through_exact_vector_for_37_rows() -> None:
    flows = {
        "flow-a": make_flow("flow-a", "A"),
        "flow-b": make_flow("flow-b", "B"),
    }
    axis_values = make_location_axis_values(37, start=10)
    category = make_category(
        {
            "flow-a": make_location_axis_candidates(flow_id="flow-a", flow_name="A", location_values=axis_values),
            "flow-b": [make_candidate(flow_id="flow-b", flow_name="B", cf_value=3.0)],
        }
    )
    matrix, _candidate_indices, _exchange_keys, _characterised, _resolved_mask, row_metadata, _stats = build_single_category_matrix(
        exchanges=[
            make_exchange("flow-a", "A", 1.0),
            make_exchange("flow-b", "B", 1.0),
        ],
        flows=flows,
        category=category,
        process_data={"@id": "process-1", "name": "Exact Vector Repeat Process", "location": {"name": "CH"}},
    )
    assert matrix.shape == (37, 2)
    assert all(row.scenario_type == "regional_cf" for row in row_metadata)
    assert all(float(row[1]) == 3.0 for row in matrix)


def test_effect_equivalent_candidates_collapse_before_candidate_set_limit() -> None:
    flows = {
        "flow-a": make_flow("flow-a", "A"),
    }
    category = make_category(
        {
            "flow-a": [
                make_candidate(flow_id="flow-a", flow_name="A", cf_value=10.0),
                make_candidate(flow_id="flow-a", flow_name="A", cf_value=10.0),
                make_candidate(flow_id="flow-a", flow_name="A", cf_value=10.0),
            ],
        }
    )
    matrix, _candidate_indices, _exchange_keys, _characterised, _resolved_mask, row_metadata, stats = build_single_category_matrix(
        exchanges=[make_exchange("flow-a", "A", 2.0)],
        flows=flows,
        category=category,
        process_data={"@id": "process-1", "name": "Deduped Process"},
        max_candidate_set_size=1,
    )
    assert matrix.shape == (1, 1)
    assert np.allclose(matrix[0], np.array([20.0]))
    assert row_metadata[0].scenario_type == "exact"
    assert stats.n_finite_cf_candidate_sets == 0


def test_regional_duplicate_candidates_do_not_create_cartesian_duplicates() -> None:
    flows = {
        "flow-a": make_flow("flow-a", "A"),
        "flow-b": make_flow("flow-b", "B"),
    }
    category = make_category(
        {
            "flow-a": [
                make_candidate(flow_id="flow-a", flow_name="A", cf_value=10.0, cf_location_id="GB", cf_location_name="GB"),
                make_candidate(flow_id="flow-a", flow_name="A", cf_value=10.0, cf_location_id="GB", cf_location_name="GB"),
                make_candidate(flow_id="flow-a", flow_name="A", cf_value=100.0, cf_location_id="US", cf_location_name="US"),
            ],
            "flow-b": [
                make_candidate(flow_id="flow-b", flow_name="B", cf_value=1.0, cf_location_id="GB", cf_location_name="GB"),
                make_candidate(flow_id="flow-b", flow_name="B", cf_value=1.0, cf_location_id="GB", cf_location_name="GB"),
                make_candidate(flow_id="flow-b", flow_name="B", cf_value=50.0, cf_location_id="US", cf_location_name="US"),
            ],
        }
    )
    matrix, _candidate_indices, _exchange_keys, _characterised, _resolved_mask, row_metadata, _stats = build_single_category_matrix(
        exchanges=[
            make_exchange("flow-a", "A", 1.0),
            make_exchange("flow-b", "B", 1.0),
        ],
        flows=flows,
        category=category,
        process_data={"@id": "process-1", "name": "Regional Deduped Process", "location": {"name": "EU"}},
    )
    assert matrix.shape == (2, 2)
    assert all(row.scenario_type == "regional_cf" for row in row_metadata)
    assert np.allclose(matrix, np.array([[10.0, 1.0], [100.0, 50.0]]))


def test_nonregional_finite_ambiguity_stays_independent() -> None:
    flows = {
        "flow-a": make_flow("flow-a", "A"),
        "flow-b": make_flow("flow-b", "B"),
    }
    category = make_category(
        {
            "flow-a": [
                make_candidate(flow_id="flow-a", flow_name="A", cf_value=10.0),
                make_candidate(flow_id="flow-a", flow_name="A", cf_value=20.0),
            ],
            "flow-b": [
                make_candidate(flow_id="flow-b", flow_name="B", cf_value=1.0),
                make_candidate(flow_id="flow-b", flow_name="B", cf_value=2.0),
            ],
        }
    )
    matrix, _candidate_indices, _exchange_keys, _characterised, _resolved_mask, row_metadata, stats = build_single_category_matrix(
        exchanges=[
            make_exchange("flow-a", "A", 1.0),
            make_exchange("flow-b", "B", 1.0),
        ],
        flows=flows,
        category=category,
        process_data={"@id": "process-1", "name": "Independent Finite Process"},
    )
    observed_rows = {tuple(row.tolist()) for row in matrix}
    assert matrix.shape == (4, 2)
    assert observed_rows == {(10.0, 1.0), (10.0, 2.0), (20.0, 1.0), (20.0, 2.0)}
    assert all(row.scenario_type == "finite_cf" for row in row_metadata)
    assert stats.n_regional_scenario_groups == 0
    assert stats.n_independent_candidate_groups == 2


def test_same_location_with_two_effective_values_stays_regional() -> None:
    flows = {
        "flow-a": make_flow("flow-a", "A"),
    }
    category = make_category(
        {
            "flow-a": [
                make_candidate(flow_id="flow-a", flow_name="A", cf_value=10.0, cf_location_id="GB", cf_location_name="GB"),
                make_candidate(flow_id="flow-a", flow_name="A", cf_value=11.0, cf_location_id="GB", cf_location_name="GB"),
                make_candidate(flow_id="flow-a", flow_name="A", cf_value=100.0, cf_location_id="US", cf_location_name="US"),
            ],
        }
    )
    matrix, _candidate_indices, _exchange_keys, _characterised, _resolved_mask, row_metadata, stats = build_single_category_matrix(
        exchanges=[make_exchange("flow-a", "A", 1.0)],
        flows=flows,
        category=category,
    )
    assert matrix.shape == (3, 1)
    assert {tuple(row.tolist()) for row in matrix} == {(10.0,), (11.0,), (100.0,)}
    assert all(row.scenario_type == "regional_cf" for row in row_metadata)
    assert stats.n_regional_scenario_groups == 1
    assert stats.n_independent_candidate_groups == 0


def test_within_location_multiplicity_only_expands_that_location_row() -> None:
    flows = {
        "flow-a": make_flow("flow-a", "A"),
        "flow-b": make_flow("flow-b", "B"),
    }
    # Scenario rows are keyed by real CF locations. Only local alternatives inside
    # the same location row are multiplied; generic fallbacks never create rows and
    # true non-location ambiguity stays in independent finite groups.
    category = make_category(
        {
            "flow-a": [
                make_candidate(flow_id="flow-a", flow_name="A", cf_value=10.0, cf_location_id="GB", cf_location_name="GB"),
                make_candidate(flow_id="flow-a", flow_name="A", cf_value=11.0, cf_location_id="GB", cf_location_name="GB"),
                make_candidate(flow_id="flow-a", flow_name="A", cf_value=100.0, cf_location_id="US", cf_location_name="US"),
            ],
            "flow-b": [
                make_candidate(flow_id="flow-b", flow_name="B", cf_value=1.0, cf_location_id="GB", cf_location_name="GB"),
                make_candidate(flow_id="flow-b", flow_name="B", cf_value=2.0, cf_location_id="GB", cf_location_name="GB"),
                make_candidate(flow_id="flow-b", flow_name="B", cf_value=3.0, cf_location_id="GB", cf_location_name="GB"),
                make_candidate(flow_id="flow-b", flow_name="B", cf_value=50.0, cf_location_id="US", cf_location_name="US"),
            ],
        }
    )
    matrix, _candidate_indices, _exchange_keys, _characterised, _resolved_mask, row_metadata, stats = build_single_category_matrix(
        exchanges=[
            make_exchange("flow-a", "A", 1.0),
            make_exchange("flow-b", "B", 1.0),
        ],
        flows=flows,
        category=category,
    )
    assert matrix.shape == (7, 2)
    assert {tuple(row.tolist()) for row in matrix} == {
        (10.0, 1.0),
        (10.0, 2.0),
        (10.0, 3.0),
        (11.0, 1.0),
        (11.0, 2.0),
        (11.0, 3.0),
        (100.0, 50.0),
    }
    assert all(row.scenario_type == "regional_cf" for row in row_metadata)
    assert stats.n_regional_scenario_groups == 1
    assert stats.n_independent_candidate_groups == 0


def test_37_location_axis_with_one_local_duplicate_stays_shared() -> None:
    flows = {
        "flow-a": make_flow("flow-a", "A"),
        "flow-b": make_flow("flow-b", "B"),
    }
    axis_values = make_location_axis_values(37, start=10)
    duplicated_axis_values = [*axis_values, ("L05", 999.0)]
    category = make_category(
        {
            "flow-a": make_location_axis_candidates(
                flow_id="flow-a",
                flow_name="A",
                location_values=duplicated_axis_values,
            ),
            "flow-b": make_location_axis_candidates(flow_id="flow-b", flow_name="B", location_values=axis_values),
        }
    )
    matrix, _candidate_indices, _exchange_keys, _characterised, _resolved_mask, row_metadata, stats = build_single_category_matrix(
        exchanges=[
            make_exchange("flow-a", "A", 1.0),
            make_exchange("flow-b", "B", 1.0),
        ],
        flows=flows,
        category=category,
    )
    assert matrix.shape == (38, 2)
    assert sum("L05" in row.scenario_label for row in row_metadata) == 2
    assert all(row.scenario_type == "regional_cf" for row in row_metadata)
    assert stats.n_regional_scenario_groups == 1
    assert stats.n_independent_candidate_groups == 0


def test_multiple_generic_values_without_locations_stays_finite_nonlocation() -> None:
    flows = {
        "flow-a": make_flow("flow-a", "A"),
    }
    category = make_category(
        {
            "flow-a": [
                make_candidate(flow_id="flow-a", flow_name="A", cf_value=10.0),
                make_candidate(flow_id="flow-a", flow_name="A", cf_value=20.0),
            ],
        }
    )
    matrix, _candidate_indices, _exchange_keys, _characterised, _resolved_mask, row_metadata, stats = build_single_category_matrix(
        exchanges=[make_exchange("flow-a", "A", 1.0)],
        flows=flows,
        category=category,
    )
    assert matrix.shape == (2, 1)
    assert all(row.scenario_type == "finite_cf" for row in row_metadata)
    assert stats.n_regional_scenario_groups == 0
    assert stats.n_independent_candidate_groups == 1


def test_large_generic_finite_groups_compress_to_signed_extremes() -> None:
    flows = {
        "flow-a": make_flow("flow-a", "A"),
        "flow-b": make_flow("flow-b", "B"),
        "flow-c": make_flow("flow-c", "C"),
    }
    category = make_category(
        {
            "flow-a": [
                make_candidate(flow_id="flow-a", flow_name="A", cf_value=float(value))
                for value in range(1, 38)
            ],
            "flow-b": [
                make_candidate(flow_id="flow-b", flow_name="B", cf_value=float(value))
                for value in range(10, 47)
            ],
            "flow-c": [
                make_candidate(flow_id="flow-c", flow_name="C", cf_value=float(value))
                for value in range(100, 136)
            ],
        }
    )
    matrix, _candidate_indices, _exchange_keys, _characterised, _resolved_mask, row_metadata, stats = build_single_category_matrix(
        exchanges=[
            make_exchange("flow-a", "A", 1.0),
            make_exchange("flow-b", "B", 1.0),
            make_exchange("flow-c", "C", 1.0),
        ],
        flows=flows,
        category=category,
        max_scenario_rows_per_process=10,
    )
    assert matrix.shape == (8, 3)
    assert all(row.scenario_type == "finite_cf" for row in row_metadata)
    assert stats.n_independent_candidate_groups == 3


def test_many_binary_finite_groups_fit_under_default_limit() -> None:
    flows = {
        f"flow-{index}": make_flow(f"flow-{index}", f"Flow {index}")
        for index in range(16)
    }
    category = make_category(
        {
            flow_id: [
                make_candidate(flow_id=flow_id, flow_name=flow.name, cf_value=1.0),
                make_candidate(flow_id=flow_id, flow_name=flow.name, cf_value=2.0),
            ]
            for flow_id, flow in flows.items()
        }
    )
    matrix, _candidate_indices, _exchange_keys, _characterised, _resolved_mask, row_metadata, stats = build_single_category_matrix(
        exchanges=[
            make_exchange(flow_id, flow.name, 1.0)
            for flow_id, flow in flows.items()
        ],
        flows=flows,
        category=category,
    )
    assert matrix.shape == (65536, 16)
    assert all(row.scenario_type == "finite_cf" for row in row_metadata)
    assert stats.n_independent_candidate_groups == 16


def test_non_nested_location_axes_with_generic_fallback_unify_into_one_table() -> None:
    flows = {
        "flow-a": make_flow("flow-a", "A"),
        "flow-b": make_flow("flow-b", "B"),
        "flow-c": make_flow("flow-c", "C"),
    }
    category = make_category(
        {
            "flow-a": make_location_axis_candidates(
                flow_id="flow-a",
                flow_name="A",
                location_values=[("GB", 10.0), ("US", 20.0)],
                generic_values=[900.0],
            ),
            "flow-b": make_location_axis_candidates(
                flow_id="flow-b",
                flow_name="B",
                location_values=[("US", 2.0), ("CA", 3.0)],
                generic_values=[800.0],
            ),
            "flow-c": make_location_axis_candidates(
                flow_id="flow-c",
                flow_name="C",
                location_values=[("GB", 1.0), ("FR", 4.0)],
                generic_values=[700.0],
            ),
        }
    )
    matrix, _candidate_indices, _exchange_keys, _characterised, _resolved_mask, row_metadata, stats = build_single_category_matrix(
        exchanges=[
            make_exchange("flow-a", "A", 1.0),
            make_exchange("flow-b", "B", 1.0),
            make_exchange("flow-c", "C", 1.0),
        ],
        flows=flows,
        category=category,
    )
    assert matrix.shape == (4, 3)
    assert {tuple(row.tolist()) for row in matrix} == {
        (10.0, 800.0, 1.0),
        (20.0, 2.0, 700.0),
        (900.0, 3.0, 700.0),
        (900.0, 800.0, 4.0),
    }
    assert stats.n_regional_scenario_groups == 1
    assert stats.n_independent_candidate_groups == 0
    assert all(row.scenario_type == "regional_cf" for row in row_metadata)
    assert not any("generic/unlocated" in row.scenario_label for row in row_metadata)


def test_non_nested_location_axes_without_fallback_stay_sparse_not_fatal() -> None:
    flows = {
        "flow-a": make_flow("flow-a", "A"),
        "flow-b": make_flow("flow-b", "B"),
        "flow-c": make_flow("flow-c", "C"),
    }
    category = make_category(
        {
            "flow-a": make_location_axis_candidates(
                flow_id="flow-a",
                flow_name="A",
                location_values=[("Yemen", 10.0), ("Tonga", 20.0)],
            ),
            "flow-b": make_location_axis_candidates(
                flow_id="flow-b",
                flow_name="B",
                location_values=[("Kuwait", 1.0), ("Australia and New Zealand", 2.0)],
            ),
            "flow-c": make_location_axis_candidates(
                flow_id="flow-c",
                flow_name="C",
                location_values=[("Georgia", 5.0), ("Africa", 6.0)],
            ),
        }
    )
    matrix, _candidate_indices, _exchange_keys, _characterised, _resolved_mask, row_metadata, stats = build_single_category_matrix(
        exchanges=[
            make_exchange("flow-a", "A", 1.0),
            make_exchange("flow-b", "B", 1.0),
            make_exchange("flow-c", "C", 1.0),
        ],
        flows=flows,
        category=category,
        process_data={"@id": "process-1", "name": "Polystyrene", "location": {"name": "Europe"}},
    )
    assert matrix.shape == (6, 3)
    assert {tuple(row.tolist()) for row in matrix} == {
        (10.0, 0.0, 0.0),
        (20.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 2.0, 0.0),
        (0.0, 0.0, 5.0),
        (0.0, 0.0, 6.0),
    }
    assert stats.n_regional_scenario_groups == 1
    assert stats.n_independent_candidate_groups == 0
    assert all(row.scenario_type == "regional_cf" for row in row_metadata)
    assert any("not_applicable" in row.scenario_label for row in row_metadata)
    assert not any("process:Europe" in row.scenario_label for row in row_metadata)


def test_missing_location_without_fallback_is_absent_not_fatal() -> None:
    flows = {
        "flow-a": make_flow("flow-a", "A"),
        "flow-b": make_flow("flow-b", "B"),
    }
    category = make_category(
        {
            "flow-a": make_location_axis_candidates(
                flow_id="flow-a",
                flow_name="A",
                location_values=[("Yemen", 10.0), ("Tonga", 20.0)],
            ),
            "flow-b": make_location_axis_candidates(
                flow_id="flow-b",
                flow_name="B",
                location_values=[("Yemen", 1.0), ("Kuwait", 2.0)],
            ),
        }
    )
    matrix, _candidate_indices, _exchange_keys, _characterised, _resolved_mask, row_metadata, _stats = build_single_category_matrix(
        exchanges=[
            make_exchange("flow-a", "A", 1.0),
            make_exchange("flow-b", "B", 1.0),
        ],
        flows=flows,
        category=category,
    )
    assert matrix.shape == (3, 2)
    assert {tuple(row.tolist()) for row in matrix} == {
        (10.0, 1.0),
        (20.0, 0.0),
        (0.0, 2.0),
    }
    tonga_rows = [row for row, meta in zip(matrix, row_metadata) if "=Tonga" in meta.scenario_label]
    assert len(tonga_rows) == 1
    assert tuple(tonga_rows[0].tolist()) == (20.0, 0.0)


def test_missing_location_with_fallback_uses_fallback_in_real_row() -> None:
    flows = {
        "flow-a": make_flow("flow-a", "A"),
        "flow-b": make_flow("flow-b", "B"),
    }
    category = make_category(
        {
            "flow-a": make_location_axis_candidates(
                flow_id="flow-a",
                flow_name="A",
                location_values=[("Yemen", 10.0), ("Tonga", 20.0)],
            ),
            "flow-b": make_location_axis_candidates(
                flow_id="flow-b",
                flow_name="B",
                location_values=[("Yemen", 1.0), ("Kuwait", 2.0)],
                generic_values=[500.0],
            ),
        }
    )
    matrix, _candidate_indices, _exchange_keys, _characterised, _resolved_mask, row_metadata, _stats = build_single_category_matrix(
        exchanges=[
            make_exchange("flow-a", "A", 1.0),
            make_exchange("flow-b", "B", 1.0),
        ],
        flows=flows,
        category=category,
    )
    assert matrix.shape == (3, 2)
    assert {tuple(row.tolist()) for row in matrix} == {
        (10.0, 1.0),
        (20.0, 500.0),
        (0.0, 2.0),
    }
    assert any("fallback_for_shared_region=yes" in row.scenario_label for row in row_metadata)


def test_exact_location_match_beats_authorised_parent_fallback() -> None:
    flows = {
        "flow-a": make_flow("flow-a", "A"),
    }
    category = make_category(
        {
            "flow-a": [
                make_candidate(
                    flow_id="flow-a",
                    flow_name="A",
                    cf_value=20.0,
                    cf_location_id="Tonga",
                    cf_location_name="Tonga",
                ),
                make_candidate(
                    flow_id="flow-a",
                    flow_name="A",
                    cf_value=200.0,
                    cf_location_id="Oceania",
                    cf_location_name="Oceania",
                    raw_factor_object={"value": 200.0, "locationCoverage": ["Tonga", "Niue"]},
                ),
            ],
        }
    )
    matrix, _candidate_indices, _exchange_keys, _characterised, _resolved_mask, row_metadata, _stats = build_single_category_matrix(
        exchanges=[make_exchange("flow-a", "A", 1.0)],
        flows=flows,
        category=category,
    )
    tonga_rows = [row for row, meta in zip(matrix, row_metadata) if "=Tonga" in meta.scenario_label]
    assert len(tonga_rows) == 1
    assert tuple(tonga_rows[0].tolist()) == (20.0,)
    assert not any("fallback_for_case=authorised_parent_region" in meta.scenario_label and "=Tonga" in meta.scenario_label for meta in row_metadata)


def test_authorised_parent_region_fallback_is_used_when_exact_missing() -> None:
    flows = {
        "flow-a": make_flow("flow-a", "A"),
        "flow-b": make_flow("flow-b", "B"),
    }
    category = make_category(
        {
            "flow-a": [
                make_candidate(
                    flow_id="flow-a",
                    flow_name="A",
                    cf_value=200.0,
                    cf_location_id="Oceania",
                    cf_location_name="Oceania",
                    raw_factor_object={"value": 200.0, "locationCoverage": ["Tonga", "Niue"]},
                ),
                make_candidate(
                    flow_id="flow-a",
                    flow_name="A",
                    cf_value=50.0,
                    cf_location_id="Europe",
                    cf_location_name="Europe",
                    raw_factor_object={"value": 50.0, "locationCoverage": ["Switzerland"]},
                ),
                make_candidate(flow_id="flow-a", flow_name="A", cf_value=900.0),
            ],
            "flow-b": [
                make_candidate(
                    flow_id="flow-b",
                    flow_name="B",
                    cf_value=1.0,
                    cf_location_id="Tonga",
                    cf_location_name="Tonga",
                ),
                make_candidate(
                    flow_id="flow-b",
                    flow_name="B",
                    cf_value=2.0,
                    cf_location_id="US",
                    cf_location_name="US",
                ),
            ],
        }
    )
    matrix, _candidate_indices, _exchange_keys, _characterised, _resolved_mask, row_metadata, _stats = build_single_category_matrix(
        exchanges=[make_exchange("flow-a", "A", 1.0), make_exchange("flow-b", "B", 1.0)],
        flows=flows,
        category=category,
    )
    tonga_rows = [row for row, meta in zip(matrix, row_metadata) if "=Tonga" in meta.scenario_label]
    assert len(tonga_rows) == 1
    assert tuple(tonga_rows[0].tolist()) == (200.0, 1.0)
    assert any("fallback_for_case=authorised_parent_region" in meta.scenario_label for meta in row_metadata if "=Tonga" in meta.scenario_label)


def test_generic_fallback_used_only_when_no_exact_or_authorised_parent_exists() -> None:
    flows = {
        "flow-a": make_flow("flow-a", "A"),
        "flow-b": make_flow("flow-b", "B"),
    }
    category = make_category(
        {
            "flow-a": [
                make_candidate(
                    flow_id="flow-a",
                    flow_name="A",
                    cf_value=200.0,
                    cf_location_id="Oceania",
                    cf_location_name="Oceania",
                    raw_factor_object={"value": 200.0, "locationCoverage": ["Tonga"]},
                ),
                make_candidate(
                    flow_id="flow-a",
                    flow_name="A",
                    cf_value=50.0,
                    cf_location_id="Europe",
                    cf_location_name="Europe",
                    raw_factor_object={"value": 50.0, "locationCoverage": ["Switzerland"]},
                ),
                make_candidate(flow_id="flow-a", flow_name="A", cf_value=900.0),
            ],
            "flow-b": [
                make_candidate(
                    flow_id="flow-b",
                    flow_name="B",
                    cf_value=1.0,
                    cf_location_id="Niue",
                    cf_location_name="Niue",
                ),
                make_candidate(
                    flow_id="flow-b",
                    flow_name="B",
                    cf_value=2.0,
                    cf_location_id="US",
                    cf_location_name="US",
                ),
            ],
        }
    )
    matrix, _candidate_indices, _exchange_keys, _characterised, _resolved_mask, row_metadata, _stats = build_single_category_matrix(
        exchanges=[make_exchange("flow-a", "A", 1.0), make_exchange("flow-b", "B", 1.0)],
        flows=flows,
        category=category,
    )
    niue_rows = [row for row, meta in zip(matrix, row_metadata) if "=Niue" in meta.scenario_label]
    assert len(niue_rows) == 1
    assert tuple(niue_rows[0].tolist()) == (900.0, 1.0)
    assert any("fallback_for_shared_region=yes" in meta.scenario_label for meta in row_metadata if "=Niue" in meta.scenario_label)


def test_multiple_equally_valid_parent_fallbacks_expand_as_fallback_ambiguity() -> None:
    flows = {
        "flow-a": make_flow("flow-a", "A"),
        "flow-b": make_flow("flow-b", "B"),
    }
    category = make_category(
        {
            "flow-a": [
                make_candidate(
                    flow_id="flow-a",
                    flow_name="A",
                    cf_value=200.0,
                    cf_location_id="Oceania-A",
                    cf_location_name="Oceania-A",
                    raw_factor_object={"value": 200.0, "locationCoverage": ["Tonga", "Niue"]},
                ),
                make_candidate(
                    flow_id="flow-a",
                    flow_name="A",
                    cf_value=300.0,
                    cf_location_id="Oceania-B",
                    cf_location_name="Oceania-B",
                    raw_factor_object={"value": 300.0, "locationCoverage": ["Tonga", "Niue"]},
                ),
            ],
            "flow-b": [
                make_candidate(
                    flow_id="flow-b",
                    flow_name="B",
                    cf_value=1.0,
                    cf_location_id="Tonga",
                    cf_location_name="Tonga",
                ),
                make_candidate(
                    flow_id="flow-b",
                    flow_name="B",
                    cf_value=2.0,
                    cf_location_id="US",
                    cf_location_name="US",
                ),
            ],
        }
    )
    matrix, _candidate_indices, _exchange_keys, _characterised, _resolved_mask, row_metadata, _stats = build_single_category_matrix(
        exchanges=[make_exchange("flow-a", "A", 1.0), make_exchange("flow-b", "B", 1.0)],
        flows=flows,
        category=category,
    )
    tonga_rows = [tuple(row.tolist()) for row, meta in zip(matrix, row_metadata) if "=Tonga" in meta.scenario_label]
    assert set(tonga_rows) == {(200.0, 1.0), (300.0, 1.0)}
    assert all("fallback_for_case=authorised_parent_region" in meta.scenario_label for meta in row_metadata if "=Tonga" in meta.scenario_label)


def test_excessive_scenario_expansion_fails_clearly() -> None:
    flows = {
        "flow-exact": make_flow("flow-exact", "Exact"),
        "flow-amb": make_flow("flow-amb", "Ambiguous"),
    }
    category = make_category(
        {
            "flow-exact": [make_candidate(flow_id="flow-exact", flow_name="Exact", cf_value=1.0)],
            "flow-amb": [
                make_candidate(flow_id="flow-amb", flow_name="Ambiguous", cf_value=20.0),
                make_candidate(flow_id="flow-amb", flow_name="Ambiguous", cf_value=100.0),
            ],
        }
    )
    with pytest.raises(ScenarioExpansionError, match="max_scenario_rows_per_process"):
        build_single_category_matrix(
            exchanges=[
                make_exchange("flow-exact", "Exact", 80.0),
                make_exchange("flow-amb", "Ambiguous", 1.0),
            ],
            flows=flows,
            category=category,
            process_data={"@id": "process-1", "name": "Scenario Process"},
            max_scenario_rows_per_process=1,
        )


def test_excessive_scenario_expansion_reports_group_diagnostics() -> None:
    flows = {
        "flow-a": make_flow("flow-a", "A"),
        "flow-b": make_flow("flow-b", "B"),
    }
    category = make_category(
        {
            "flow-a": [
                make_candidate(flow_id="flow-a", flow_name="A", cf_value=10.0),
                make_candidate(flow_id="flow-a", flow_name="A", cf_value=20.0),
            ],
            "flow-b": [
                make_candidate(flow_id="flow-b", flow_name="B", cf_value=1.0),
                make_candidate(flow_id="flow-b", flow_name="B", cf_value=2.0),
            ],
        }
    )
    with pytest.raises(ScenarioExpansionError) as excinfo:
        build_single_category_matrix(
            exchanges=[
                make_exchange("flow-a", "A", 3.0),
                make_exchange("flow-b", "B", 4.0),
            ],
            flows=flows,
            category=category,
            process_data={"@id": "process-1", "name": "Diagnostic Process"},
            max_scenario_rows_per_process=3,
        )
    message = str(excinfo.value)
    assert "current_group=exchange:" in message
    assert "projected_scenario_rows=4" in message
    assert "amount_range=[4, 4]" in message or "amount_range=[3, 3]" in message
    assert "contribution_range=" in message


def test_excessive_within_location_ambiguity_reports_location_specific_diagnostic() -> None:
    flows = {
        "flow-a": make_flow("flow-a", "A"),
        "flow-b": make_flow("flow-b", "B"),
    }
    category = make_category(
        {
            "flow-a": [
                make_candidate(flow_id="flow-a", flow_name="A", cf_value=10.0, cf_location_id="GB", cf_location_name="GB"),
                make_candidate(flow_id="flow-a", flow_name="A", cf_value=11.0, cf_location_id="GB", cf_location_name="GB"),
                make_candidate(flow_id="flow-a", flow_name="A", cf_value=100.0, cf_location_id="US", cf_location_name="US"),
            ],
            "flow-b": [
                make_candidate(flow_id="flow-b", flow_name="B", cf_value=1.0, cf_location_id="GB", cf_location_name="GB"),
                make_candidate(flow_id="flow-b", flow_name="B", cf_value=2.0, cf_location_id="GB", cf_location_name="GB"),
                make_candidate(flow_id="flow-b", flow_name="B", cf_value=50.0, cf_location_id="US", cf_location_name="US"),
            ],
        }
    )
    with pytest.raises(ScenarioExpansionError) as excinfo:
        build_single_category_matrix(
            exchanges=[
                make_exchange("flow-a", "A", 1.0),
                make_exchange("flow-b", "B", 1.0),
            ],
            flows=flows,
            category=category,
            max_scenario_rows_per_process=3,
        )
    message = str(excinfo.value)
    assert "excessive within-case ambiguity" in message
    assert "scenario_location=GB" in message
    assert "current_group=location_axis:" in message
