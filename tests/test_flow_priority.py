import csv
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

from lci_reduce.cli import priority_command
from lci_reduce.flow_priority import build_greedy_ladder, prefix_length_for_tau, single_flow_shortfall


def _write_archive(path: Path, files: dict[str, dict]) -> Path:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for rel_path, data in files.items():
            archive.writestr(rel_path, json.dumps(data))
    return path


def make_priority_database_zip(base: Path, name: str = "priority_db.zip") -> Path:
    return _write_archive(
        base / name,
        {
            "processes/process-1.json": {
                "@id": "process-1",
                "@type": "Process",
                "name": "Priority Process",
                "exchanges": [
                    {
                        "@id": "product",
                        "amount": 1.0,
                        "flow": {"@id": "flow-product", "name": "Product"},
                        "unit": {"name": "kg"},
                        "quantitativeReference": True,
                    },
                    {
                        "@id": "co2-1",
                        "amount": 10.0,
                        "flow": {"@id": "flow-co2", "name": "CO2"},
                        "unit": {"name": "kg"},
                    },
                    {
                        "@id": "unk-1",
                        "amount": 5.0,
                        "flow": {"@id": "flow-unk", "name": "Unmapped elementary flow"},
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
            "flows/flow-co2.json": {
                "@id": "flow-co2",
                "@type": "Flow",
                "name": "CO2",
                "flowType": "ELEMENTARY_FLOW",
                "categoryPath": "air/urban air",
            },
            "flows/flow-unk.json": {
                "@id": "flow-unk",
                "@type": "Flow",
                "name": "Unmapped elementary flow",
                "flowType": "ELEMENTARY_FLOW",
                "categoryPath": "air/urban air",
            },
            "lcia_methods/method-1.json": {
                "@id": "method-1",
                "@type": "ImpactMethod",
                "name": "IPCC 2021",
            },
            "lcia_categories/category-1.json": {
                "@id": "category-1",
                "@type": "ImpactCategory",
                "name": "Climate change",
                "impactMethod": {"@id": "method-1", "name": "IPCC 2021"},
                "referenceUnitName": "kg",
                "impactFactors": [
                    {"flow": {"@id": "flow-co2", "name": "CO2"}, "value": 1.0, "unitName": "kg"},
                ],
            },
        },
    )


def make_priority_methods_zip(base: Path, name: str = "priority_methods.zip") -> Path:
    return _write_archive(
        base / name,
        {
            "external_methods/method-ext.json": {
                "@id": "method-ext",
                "@type": "ImpactMethod",
                "name": "External Priority Method",
            },
            "external_categories/category-1.json": {
                "@id": "category-1",
                "@type": "ImpactCategory",
                "name": "Climate change",
                "impactMethod": {"@id": "method-ext", "name": "External Priority Method"},
                "referenceUnitName": "kg",
                "impactFactors": [
                    {"flow": {"@id": "flow-co2", "name": "CO2"}, "value": 3.5, "unitName": "kg"},
                ],
            },
        },
    )


def make_priority_database_with_category_refs_zip(base: Path, name: str = "priority_db_category_refs.zip") -> Path:
    return _write_archive(
        base / name,
        {
            "processes/process-1.json": {
                "@id": "process-1",
                "@type": "Process",
                "name": "Priority Process",
                "exchanges": [
                    {
                        "@id": "product",
                        "amount": 1.0,
                        "flow": {"@id": "flow-product", "name": "Product"},
                        "unit": {"name": "kg"},
                        "quantitativeReference": True,
                    },
                    {
                        "@id": "co2-1",
                        "amount": 10.0,
                        "flow": {"@id": "flow-co2", "name": "CO2"},
                        "unit": {"name": "kg"},
                    },
                ],
            },
            "flows/flow-product.json": {
                "@id": "flow-product",
                "@type": "Flow",
                "name": "Product",
                "flowType": "PRODUCT_FLOW",
                "category": {"@id": "cat-products", "name": "products"},
            },
            "flows/flow-co2.json": {
                "@id": "flow-co2",
                "@type": "Flow",
                "name": "CO2",
                "flowType": "ELEMENTARY_FLOW",
                "category": {"@id": "cat-urban-air", "name": "urban air"},
            },
            "categories/cat-products.json": {
                "@id": "cat-products",
                "@type": "Category",
                "name": "products",
            },
            "categories/cat-air.json": {
                "@id": "cat-air",
                "@type": "Category",
                "name": "air",
            },
            "categories/cat-urban-air.json": {
                "@id": "cat-urban-air",
                "@type": "Category",
                "name": "urban air",
                "category": {"@id": "cat-air", "name": "air"},
            },
            "lcia_methods/method-1.json": {
                "@id": "method-1",
                "@type": "ImpactMethod",
                "name": "IPCC 2021",
            },
            "lcia_categories/category-1.json": {
                "@id": "category-1",
                "@type": "ImpactCategory",
                "name": "Climate change",
                "impactMethod": {"@id": "method-1", "name": "IPCC 2021"},
                "referenceUnitName": "kg",
                "impactFactors": [
                    {"flow": {"@id": "flow-co2", "name": "CO2"}, "value": 1.0, "unitName": "kg"},
                ],
            },
        },
    )


def make_priority_witness_database_zip(base: Path, name: str = "priority_witness_db.zip") -> Path:
    return _write_archive(
        base / name,
        {
            "processes/process-pos.json": {
                "@id": "process-pos",
                "@type": "Process",
                "name": "Positive Process",
                "exchanges": [
                    {
                        "@id": "product-pos",
                        "amount": 1.0,
                        "flow": {"@id": "flow-product", "name": "Product"},
                        "unit": {"name": "kg"},
                        "quantitativeReference": True,
                    },
                    {
                        "@id": "shared-pos",
                        "amount": 60.0,
                        "flow": {"@id": "flow-shared", "name": "Shared flow"},
                        "unit": {"name": "kg"},
                    },
                    {
                        "@id": "plus-only",
                        "amount": 35.0,
                        "flow": {"@id": "flow-plus", "name": "Positive only flow"},
                        "unit": {"name": "kg"},
                    },
                    {
                        "@id": "zero-pos",
                        "amount": 5.0,
                        "flow": {"@id": "flow-zero", "name": "Zero eta flow"},
                        "unit": {"name": "kg"},
                    },
                ],
            },
            "processes/process-neg.json": {
                "@id": "process-neg",
                "@type": "Process",
                "name": "Negative Process",
                "exchanges": [
                    {
                        "@id": "product-neg",
                        "amount": 1.0,
                        "flow": {"@id": "flow-product", "name": "Product"},
                        "unit": {"name": "kg"},
                        "quantitativeReference": True,
                    },
                    {
                        "@id": "shared-neg",
                        "amount": -90.0,
                        "flow": {"@id": "flow-shared", "name": "Shared flow"},
                        "unit": {"name": "kg"},
                    },
                    {
                        "@id": "minus-only",
                        "amount": -10.0,
                        "flow": {"@id": "flow-minus", "name": "Negative only flow"},
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
            "flows/flow-shared.json": {
                "@id": "flow-shared",
                "@type": "Flow",
                "name": "Shared flow",
                "flowType": "ELEMENTARY_FLOW",
                "categoryPath": "air/urban air",
            },
            "flows/flow-plus.json": {
                "@id": "flow-plus",
                "@type": "Flow",
                "name": "Positive only flow",
                "flowType": "ELEMENTARY_FLOW",
                "categoryPath": "air/urban air",
            },
            "flows/flow-zero.json": {
                "@id": "flow-zero",
                "@type": "Flow",
                "name": "Zero eta flow",
                "flowType": "ELEMENTARY_FLOW",
                "categoryPath": "air/urban air",
            },
            "flows/flow-minus.json": {
                "@id": "flow-minus",
                "@type": "Flow",
                "name": "Negative only flow",
                "flowType": "ELEMENTARY_FLOW",
                "categoryPath": "water/fresh water",
            },
            "lcia_methods/method-1.json": {
                "@id": "method-1",
                "@type": "ImpactMethod",
                "name": "Witness Method",
            },
            "lcia_categories/category-pos.json": {
                "@id": "category-pos",
                "@type": "ImpactCategory",
                "name": "Positive Climate",
                "impactMethod": {"@id": "method-1", "name": "Witness Method"},
                "referenceUnitName": "kg",
                "impactFactors": [
                    {"flow": {"@id": "flow-shared", "name": "Shared flow"}, "value": 1.0, "unitName": "kg"},
                    {"flow": {"@id": "flow-plus", "name": "Positive only flow"}, "value": 1.0, "unitName": "kg"},
                    {"flow": {"@id": "flow-zero", "name": "Zero eta flow"}, "value": 1.0, "unitName": "kg"},
                ],
            },
            "lcia_categories/category-neg.json": {
                "@id": "category-neg",
                "@type": "ImpactCategory",
                "name": "Negative Toxicity",
                "impactMethod": {"@id": "method-1", "name": "Witness Method"},
                "referenceUnitName": "kg",
                "impactFactors": [
                    {"flow": {"@id": "flow-shared", "name": "Shared flow"}, "value": 1.0, "unitName": "kg"},
                    {"flow": {"@id": "flow-minus", "name": "Negative only flow"}, "value": 1.0, "unitName": "kg"},
                ],
            },
        },
    )


def make_priority_zero_cf_database_zip(base: Path, name: str = "priority_zero_cf_db.zip") -> Path:
    return _write_archive(
        base / name,
        {
            "processes/process-1.json": {
                "@id": "process-1",
                "@type": "Process",
                "name": "Zero CF Process",
                "exchanges": [
                    {
                        "@id": "product",
                        "amount": 1.0,
                        "flow": {"@id": "flow-product", "name": "Product"},
                        "unit": {"name": "kg"},
                        "quantitativeReference": True,
                    },
                    {
                        "@id": "zero-cf-1",
                        "amount": 5.0,
                        "flow": {"@id": "flow-zero-cf", "name": "Zero CF flow"},
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
            "flows/flow-zero-cf.json": {
                "@id": "flow-zero-cf",
                "@type": "Flow",
                "name": "Zero CF flow",
                "flowType": "ELEMENTARY_FLOW",
                "categoryPath": "air/urban air",
            },
            "lcia_methods/method-1.json": {
                "@id": "method-1",
                "@type": "ImpactMethod",
                "name": "Zero CF Method",
            },
            "lcia_categories/category-1.json": {
                "@id": "category-1",
                "@type": "ImpactCategory",
                "name": "Zero CF Category",
                "impactMethod": {"@id": "method-1", "name": "Zero CF Method"},
                "referenceUnitName": "kg",
                "impactFactors": [
                    {"flow": {"@id": "flow-zero-cf", "name": "Zero CF flow"}, "value": 0.0, "unitName": "kg"},
                ],
            },
        },
    )


def _read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def test_eta_and_loss_max_with_overshoot() -> None:
    tau = 0.95
    coverage = 0.98
    loss = 0.02
    assert loss == 0.02
    assert single_flow_shortfall(tau, coverage, loss) == 0.0


def test_eta_when_loss_exceeds_margin() -> None:
    tau = 0.95
    coverage = 0.98
    loss = 0.05
    assert loss == 0.05
    assert single_flow_shortfall(tau, coverage, loss) == pytest.approx(0.02)


def test_group_risk_screen_uses_loss_max_not_eta_sum() -> None:
    tau = 0.95
    coverage = 0.98
    loss_a = 0.02
    loss_b = 0.02
    eta_a = single_flow_shortfall(tau, coverage, loss_a)
    eta_b = single_flow_shortfall(tau, coverage, loss_b)
    assert eta_a == 0.0
    assert eta_b == 0.0
    assert loss_a + loss_b == 0.04
    exact_joint_eta = single_flow_shortfall(tau, coverage, loss_a + loss_b)
    assert exact_joint_eta == pytest.approx(0.01)


def test_ladder_prefixes_are_deterministic_and_nested() -> None:
    matrix = np.array(
        [
            [0.90, 0.05, 0.04, 0.01],
            [0.10, 0.15, 0.70, 0.05],
        ],
        dtype=float,
    )
    ladder = build_greedy_ladder(matrix, exchange_keys=["co2", "nox", "ch4", "so2"])
    assert ladder.order == [0, 2, 1, 3]
    assert prefix_length_for_tau(ladder, 0.95) == 3
    assert prefix_length_for_tau(ladder, 0.99) == 4
    selected_095 = ladder.order[: prefix_length_for_tau(ladder, 0.95)]
    selected_099 = ladder.order[: prefix_length_for_tau(ladder, 0.99)]
    assert selected_095 == [0, 2, 1]
    assert selected_099 == [0, 2, 1, 3]
    assert selected_095 == selected_099[: len(selected_095)]


def test_ladder_handles_small_positive_contributions_near_tolerance() -> None:
    matrix = np.array([[4e-13, 4e-13, 4e-13, 0.0]], dtype=float)
    ladder = build_greedy_ladder(matrix, exchange_keys=["a", "b", "c", "z"], tol=1e-12)
    assert ladder.order == [0, 1, 2]
    assert ladder.lambda_after.tolist() == pytest.approx([1 / 3, 2 / 3, 1.0])


def test_ladder_order_is_invariant_to_category_rescaling() -> None:
    matrix = np.array(
        [
            [90.0, 0.0, 10.0],
            [0.0, 4.0, 2.0],
        ],
        dtype=float,
    )
    scaled = matrix.copy()
    scaled[0] *= 1000.0

    ladder = build_greedy_ladder(matrix, exchange_keys=["a", "b", "c"])
    scaled_ladder = build_greedy_ladder(scaled, exchange_keys=["a", "b", "c"])

    assert ladder.order == [0, 1, 2]
    assert scaled_ladder.order == [0, 1, 2]


def test_ladder_prefers_fractional_gain_over_raw_magnitude() -> None:
    matrix = np.array(
        [
            [900.0, 0.0, 100.0],
            [0.0, 0.8, 0.0],
            [0.0, 0.8, 0.0],
        ],
        dtype=float,
    )

    ladder = build_greedy_ladder(matrix, exchange_keys=["c", "a", "b"])

    assert ladder.order == [1, 0, 2]


def test_uncharacterised_flow_has_zero_priority_metrics(tmp_path: Path) -> None:
    db_zip = make_priority_database_zip(tmp_path)
    output_dir = tmp_path / "priority_out"
    result = priority_command(
        database=str(db_zip),
        methods=None,
        output=str(output_dir),
        method_selection="all",
        audit_tau=[0.95, 0.99],
        strict_units=True,
        tolerance=1e-12,
    )

    header, rows = _read_csv_rows(Path(result.flow_priority_csv))
    assert header[-1] == "cf_status"
    row_lookup = {row["flow_id"]: row for row in rows}
    unmapped = row_lookup["flow-unk"]
    assert unmapped["cf_status"] == "uncharacterised"
    assert unmapped["characterised_occurrence_count"] == "0"
    assert unmapped["eta_0_95"] == "0"
    assert unmapped["loss_max_0_95"] == "0"
    assert unmapped["eta_0_99"] == "0"
    assert unmapped["loss_max_0_99"] == "0"
    assert unmapped["tau_entry_min"] == ""
    assert unmapped["tau_entry_median"] == ""
    assert unmapped["tau_entry_max"] == ""


def test_zero_cf_occurrence_counts_as_characterised_without_priority_metrics(tmp_path: Path) -> None:
    db_zip = make_priority_zero_cf_database_zip(tmp_path)
    output_dir = tmp_path / "priority_zero_cf_out"
    result = priority_command(
        database=str(db_zip),
        methods=None,
        output=str(output_dir),
        method_selection="all",
        audit_tau=[0.95, 0.99],
        strict_units=True,
        tolerance=1e-12,
    )

    _header, rows = _read_csv_rows(Path(result.flow_priority_csv))
    row_lookup = {row["flow_id"]: row for row in rows}
    zero_cf = row_lookup["flow-zero-cf"]
    assert zero_cf["characterised_occurrence_count"] == "1"
    assert zero_cf["tau_entry_min"] == ""
    assert zero_cf["tau_entry_median"] == ""
    assert zero_cf["tau_entry_max"] == ""
    assert zero_cf["eta_0_95"] == "0"
    assert zero_cf["loss_max_0_95"] == "0"
    assert zero_cf["eta_0_99"] == "0"
    assert zero_cf["loss_max_0_99"] == "0"


def test_priority_output_resolves_flow_category_references_to_compartments(tmp_path: Path) -> None:
    db_zip = make_priority_database_with_category_refs_zip(tmp_path)
    output_dir = tmp_path / "priority_category_ref_out"
    result = priority_command(
        database=str(db_zip),
        methods=None,
        output=str(output_dir),
        method_selection="all",
        audit_tau=[0.95, 0.99],
        strict_units=True,
        tolerance=1e-12,
    )

    _header, rows = _read_csv_rows(Path(result.flow_priority_csv))
    row_lookup = {row["flow_id"]: row for row in rows}
    assert row_lookup["flow-co2"]["compartment"] == "air"
    assert row_lookup["flow-co2"]["subcompartment"] == "urban air"


def test_default_csv_schema_is_exact(tmp_path: Path) -> None:
    db_zip = make_priority_database_zip(tmp_path)
    output_dir = tmp_path / "schema_out"
    result = priority_command(
        database=str(db_zip),
        methods=None,
        output=str(output_dir),
        method_selection="all",
        audit_tau=[0.95, 0.99],
        strict_units=True,
        tolerance=1e-12,
    )
    header, _rows = _read_csv_rows(Path(result.flow_priority_csv))
    assert header == [
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
        "eta_0_95",
        "eta_0_95_witness",
        "loss_max_0_95",
        "eta_0_99",
        "eta_0_99_witness",
        "loss_max_0_99",
        "cf_status",
    ]


def test_eta_witness_is_populated_when_eta_is_positive(tmp_path: Path) -> None:
    db_zip = make_priority_witness_database_zip(tmp_path)
    output_dir = tmp_path / "witness_positive_out"
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
    assert row_lookup["flow-plus"]["eta_0_95"] != "0"
    assert row_lookup["flow-plus"]["eta_0_95_witness"] == "Positive Process | Positive Climate | +"


def test_eta_witness_is_empty_when_eta_is_zero(tmp_path: Path) -> None:
    db_zip = make_priority_witness_database_zip(tmp_path)
    output_dir = tmp_path / "witness_zero_out"
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
    assert row_lookup["flow-zero"]["eta_0_95"] == "0"
    assert row_lookup["flow-zero"]["eta_0_95_witness"] == ""


def test_eta_witness_updates_to_later_larger_maximum_and_negative_sign(tmp_path: Path) -> None:
    db_zip = make_priority_witness_database_zip(tmp_path)
    output_dir = tmp_path / "witness_negative_out"
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
    assert float(row_lookup["flow-shared"]["eta_0_95"]) == pytest.approx(0.95)
    assert row_lookup["flow-shared"]["eta_0_95_witness"] == "Negative Process | Positive Climate | -"


def test_eta_witness_records_negative_sign_for_negative_only_flow(tmp_path: Path) -> None:
    db_zip = make_priority_witness_database_zip(tmp_path)
    output_dir = tmp_path / "witness_negative_only_out"
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
    assert row_lookup["flow-minus"]["eta_0_95"] != "0"
    assert row_lookup["flow-minus"]["eta_0_95_witness"] == "Negative Process | Negative Toxicity | -"


def test_priority_run_does_not_rewrite_database_or_create_lite_zip(tmp_path: Path) -> None:
    db_zip = make_priority_database_zip(tmp_path)
    original_bytes = db_zip.read_bytes()
    output_dir = tmp_path / "no_zip_out"
    result = priority_command(
        database=str(db_zip),
        methods=None,
        output=str(output_dir),
        method_selection="all",
        audit_tau=[0.95, 0.99],
        strict_units=True,
        tolerance=1e-12,
    )

    assert Path(result.flow_priority_csv).exists()
    assert Path(result.flow_priority_metadata_json).exists()
    assert list(output_dir.glob("*.zip")) == []
    assert db_zip.read_bytes() == original_bytes


def test_priority_output_is_deterministic_except_generated_at(tmp_path: Path) -> None:
    db_zip = make_priority_database_zip(tmp_path)
    output_a = tmp_path / "determinism_a"
    output_b = tmp_path / "determinism_b"

    result_a = priority_command(
        database=str(db_zip),
        methods=None,
        output=str(output_a),
        method_selection="all",
        audit_tau=[0.95, 0.99],
        strict_units=True,
        tolerance=1e-12,
    )
    result_b = priority_command(
        database=str(db_zip),
        methods=None,
        output=str(output_b),
        method_selection="all",
        audit_tau=[0.95, 0.99],
        strict_units=True,
        tolerance=1e-12,
    )

    csv_a = Path(result_a.flow_priority_csv).read_text(encoding="utf-8")
    csv_b = Path(result_b.flow_priority_csv).read_text(encoding="utf-8")
    assert csv_a == csv_b

    metadata_a = json.loads(Path(result_a.flow_priority_metadata_json).read_text(encoding="utf-8"))
    metadata_b = json.loads(Path(result_b.flow_priority_metadata_json).read_text(encoding="utf-8"))
    assert metadata_a["lcia_method_source"] == "database"
    assert metadata_a["internal_lcia_methods_ignored"] is False
    metadata_a.pop("generated_at", None)
    metadata_b.pop("generated_at", None)
    assert metadata_a == metadata_b


def test_priority_uses_external_methods_when_provided_and_reports_metadata(tmp_path: Path) -> None:
    db_zip = make_priority_database_zip(tmp_path)
    methods_zip = make_priority_methods_zip(tmp_path)
    output_dir = tmp_path / "priority_external_methods_out"

    result = priority_command(
        database=str(db_zip),
        methods=str(methods_zip),
        output=str(output_dir),
        method_selection="all",
        audit_tau=[0.95, 0.99],
        strict_units=True,
        tolerance=1e-12,
    )

    metadata = json.loads(Path(result.flow_priority_metadata_json).read_text(encoding="utf-8"))

    assert metadata["lcia_method_source"] == "external"
    assert metadata["internal_lcia_methods_ignored"] is True
    assert metadata["selected_methods"] == [
        {
            "method_id": "method-ext",
            "method_name": "External Priority Method",
            "method_path": "",
            "source_file": "external_methods/method-ext.json",
        }
    ]
    assert metadata["selected_impact_categories"][0]["method_id"] == "method-ext"
    assert metadata["selected_impact_categories"][0]["method_name"] == "External Priority Method"
    assert metadata["selected_impact_categories"][0]["source_file"] == "external_categories/category-1.json"


def test_priority_cli_writes_sidecars_only(tmp_path: Path) -> None:
    db_zip = make_priority_database_zip(tmp_path)
    output_dir = tmp_path / "priority_cli_out"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd())

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "lci_reduce.cli",
            "priority",
            "--database",
            str(db_zip),
            "--output",
            str(output_dir),
            "--method-selection",
            "all",
            "--audit-tau",
            "0.95",
            "0.99",
            "--strict-units",
            "true",
        ],
        cwd=Path.cwd(),
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    payload = json.loads(result.stdout)
    assert Path(payload["flow_priority_csv"]).exists()
    assert Path(payload["flow_priority_metadata_json"]).exists()
    assert list(output_dir.glob("*.zip")) == []
