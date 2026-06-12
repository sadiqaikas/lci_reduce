from __future__ import annotations

import pytest

from lca_flowkit.errors import ScenarioExpansionError, UnitResolutionError
from lca_flowkit.models import Exchange, FlowDefinition, ImpactCategory, ImpactFactorCandidate, Location, ProcessRecord, UnitDefinition
from lca_flowkit.scenarios import build_process_contribution_bundle


def _flow(flow_id: str, name: str, *, category_path: str = "air/urban air") -> FlowDefinition:
    """Create a compact elementary-flow fixture."""
    compartment, _, subcompartment = category_path.partition("/")
    return FlowDefinition(
        flow_id=flow_id,
        name=name,
        flow_type="ELEMENTARY_FLOW",
        compartment=compartment,
        subcompartment=subcompartment,
        is_elementary=True,
    )


def _exchange(flow_id: str, name: str, amount: float, *, unit_name: str = "kg", unit_id: str = "", index: int = 0) -> Exchange:
    """Create one exchange fixture with only the fields used in tests."""
    return Exchange(
        exchange_id=f"ex-{flow_id}-{index}",
        index=index,
        amount=amount,
        flow_id=flow_id,
        flow_name=name,
        unit_name=unit_name,
        unit_id=unit_id,
    )


def _category(candidates: dict[str, list[ImpactFactorCandidate]]) -> ImpactCategory:
    """Create one impact-category fixture."""
    return ImpactCategory(
        category_id="cat-1",
        name="Climate change",
        method_id="method-1",
        method_name="Method",
        reference_unit="kg",
        factors_by_flow=candidates,
    )


def _candidate(
    flow_id: str,
    flow_name: str,
    value: float,
    *,
    unit_name: str = "kg",
    unit_id: str = "",
    flow_property_id: str = "",
    location_name: str = "",
    coverage_tokens: tuple[str, ...] = (),
    source_path: str = "",
) -> ImpactFactorCandidate:
    return ImpactFactorCandidate(
        category_id="cat-1",
        category_name="Climate change",
        method_id="method-1",
        method_name="Method",
        flow_id=flow_id,
        flow_name=flow_name,
        value=value,
        unit_name=unit_name,
        unit_id=unit_id,
        flow_property_id=flow_property_id,
        location=Location(name=location_name, region=location_name),
        source_path=source_path,
        raw={"locationCoverage": list(coverage_tokens)} if coverage_tokens else {},
    )


def _process(*exchanges: Exchange, location_name: str = "") -> ProcessRecord:
    """Create one process fixture with an optional location."""
    return ProcessRecord(
        process_id="process-1",
        name="Process",
        source_path="processes/process-1.json",
        location=Location(name=location_name, region=location_name),
        exchanges=list(exchanges),
        raw={"@id": "process-1", "name": "Process"},
    )


def test_exact_cf_match_builds_exact_row() -> None:
    bundle, warnings = build_process_contribution_bundle(
        _process(_exchange("flow-a", "A", 10.0), _exchange("flow-b", "B", 5.0, index=1)),
        {
            "flow-a": _flow("flow-a", "A"),
            "flow-b": _flow("flow-b", "B"),
        },
        [
            _category(
                {
                    "flow-a": [_candidate("flow-a", "A", 2.0)],
                    "flow-b": [_candidate("flow-b", "B", 3.0)],
                }
            )
        ],
        {},
        strict_units=True,
    )
    assert not warnings
    assert bundle.matrix.shape == (1, 2)
    assert bundle.row_metadata[0].scenario_type == "exact"
    assert bundle.matrix.tolist() == [[20.0, 15.0]]


def test_strict_unit_filtering_raises() -> None:
    with pytest.raises(UnitResolutionError):
        build_process_contribution_bundle(
            _process(_exchange("flow-a", "A", 10.0, unit_name="m3")),
            {"flow-a": _flow("flow-a", "A")},
            [_category({"flow-a": [_candidate("flow-a", "A", 2.0, unit_name="kg", unit_id="unit-kg")]})],
            {},
            strict_units=True,
        )


def test_duplicate_cf_collapse_records_warning() -> None:
    bundle, warnings = build_process_contribution_bundle(
        _process(_exchange("flow-a", "A", 2.0)),
        {"flow-a": _flow("flow-a", "A")},
        [_category({"flow-a": [_candidate("flow-a", "A", 10.0), _candidate("flow-a", "A", 10.0)]})],
        {},
        strict_units=True,
    )
    assert bundle.matrix.tolist() == [[20.0]]
    assert any(item.code == "duplicate_cf_collapse" for item in warnings)


def test_default_limit_handles_many_binary_finite_cases() -> None:
    flows = {
        f"flow-{index}": _flow(f"flow-{index}", f"Flow {index}")
        for index in range(16)
    }
    process = _process(
        *[
            _exchange(flow_id, flow.name, 1.0, index=index)
            for index, (flow_id, flow) in enumerate(flows.items())
        ]
    )
    category = _category(
        {
            flow_id: [
                _candidate(flow_id, flow.name, 1.0),
                _candidate(flow_id, flow.name, 2.0),
            ]
            for flow_id, flow in flows.items()
        }
    )
    bundle, warnings = build_process_contribution_bundle(
        process,
        flows,
        [category],
        {},
        strict_units=True,
    )
    assert not warnings
    assert bundle.matrix.shape == (65536, 16)
    assert all(row.scenario_type == "finite_cf" for row in bundle.row_metadata)


def test_same_contribution_with_distinct_metadata_is_not_collapsed() -> None:
    bundle, _warnings = build_process_contribution_bundle(
        _process(_exchange("flow-a", "A", 2.0)),
        {"flow-a": _flow("flow-a", "A")},
        [
            _category(
                {
                    "flow-a": [
                        _candidate("flow-a", "A", 10.0, source_path="methods/a.json"),
                        _candidate("flow-a", "A", 10.0, source_path="methods/b.json"),
                    ]
                }
            )
        ],
        {},
        strict_units=True,
    )
    assert bundle.matrix.shape == (2, 1)
    assert bundle.matrix.tolist() == [[20.0], [20.0]]
    assert not any(item.code == "duplicate_cf_collapse" for item in _warnings)


def test_blank_location_generic_fallback_participates_in_regional_rows() -> None:
    bundle, _warnings = build_process_contribution_bundle(
        _process(_exchange("flow-a", "A", 1.0), _exchange("flow-b", "B", 1.0, index=1)),
        {
            "flow-a": _flow("flow-a", "A"),
            "flow-b": _flow("flow-b", "B"),
        },
        [
            _category(
                {
                    "flow-a": [_candidate("flow-a", "A", 3.0, location_name="GB"), _candidate("flow-a", "A", 7.0, location_name="US")],
                    "flow-b": [_candidate("flow-b", "B", 9.0)],
                }
            )
        ],
        {},
        strict_units=True,
    )
    assert bundle.matrix.shape == (2, 2)
    assert {tuple(row.tolist()) for row in bundle.matrix} == {(3.0, 9.0), (7.0, 9.0)}
    assert all(row.scenario_type == "regional_cf" for row in bundle.row_metadata)


def test_real_regional_cf_ambiguity_expands_to_regional_rows() -> None:
    bundle, _warnings = build_process_contribution_bundle(
        _process(_exchange("flow-a", "A", 1.0), _exchange("flow-b", "B", 1.0, index=1)),
        {
            "flow-a": _flow("flow-a", "A"),
            "flow-b": _flow("flow-b", "B"),
        },
        [
            _category(
                {
                    "flow-a": [_candidate("flow-a", "A", 10.0, location_name="GB"), _candidate("flow-a", "A", 100.0, location_name="US")],
                    "flow-b": [_candidate("flow-b", "B", 1.0, location_name="GB"), _candidate("flow-b", "B", 50.0, location_name="US")],
                }
            )
        ],
        {},
        strict_units=True,
    )
    assert bundle.matrix.shape == (2, 2)
    assert {tuple(row.tolist()) for row in bundle.matrix} == {(10.0, 1.0), (100.0, 50.0)}
    assert all(row.scenario_type == "regional_cf" for row in bundle.row_metadata)


def test_regional_ambiguity_raises_when_expansion_limit_is_too_small() -> None:
    with pytest.raises(ScenarioExpansionError):
        build_process_contribution_bundle(
            _process(_exchange("flow-a", "A", 1.0), _exchange("flow-b", "B", 1.0, index=1)),
            {
                "flow-a": _flow("flow-a", "A"),
                "flow-b": _flow("flow-b", "B"),
            },
            [
                _category(
                    {
                        "flow-a": [_candidate("flow-a", "A", 10.0, location_name="GB"), _candidate("flow-a", "A", 100.0, location_name="US")],
                        "flow-b": [_candidate("flow-b", "B", 1.0, location_name="GB"), _candidate("flow-b", "B", 50.0, location_name="US")],
                    }
                )
            ],
            {},
            strict_units=True,
            max_rows_per_process=1,
        )


def test_registry_unit_conversion_is_applied() -> None:
    bundle, _warnings = build_process_contribution_bundle(
        _process(_exchange("flow-a", "A", 1.0, unit_name="kg", unit_id="unit-kg")),
        {"flow-a": _flow("flow-a", "A")},
        [_category({"flow-a": [_candidate("flow-a", "A", 2.0, unit_name="g", unit_id="unit-g", flow_property_id="fp-mass")]})],
        {
            "unit-kg": UnitDefinition(unit_id="unit-kg", name="kg", group_id="mass", conversion_factor=1.0, flow_property_id="fp-mass"),
            "unit-g": UnitDefinition(unit_id="unit-g", name="g", group_id="mass", conversion_factor=0.001, flow_property_id="fp-mass"),
        },
        strict_units=True,
    )
    assert bundle.matrix.tolist() == [[2000.0]]
