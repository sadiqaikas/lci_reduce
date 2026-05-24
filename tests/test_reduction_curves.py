import csv
import json
from pathlib import Path
import zipfile

from lci_reduce.models import TauReductionRun
from lci_reduce.reduction_curves import curve_point_is_valid, extract_run_metadata, group_warnings


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "process_id",
        "process_name",
        "n_total_exchanges_before",
        "n_total_exchanges_after",
        "n_elementary_before",
        "n_elementary_after",
        "n_elementary_characterised",
        "n_elementary_uncharacterised",
        "n_uncharacterised_kept",
        "n_uncharacterised_removed",
        "n_elementary_removed",
        "n_protected_exchanges",
        "positive_cover_ok",
        "negative_cover_ok",
        "coverage_failure",
        "status",
        "warnings",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _make_toy_database_zip(base: Path, name: str = "toy_db.zip") -> Path:
    db_zip = base / name
    files = {
        "processes/process-1.json": {
            "@id": "process-1",
            "@type": "Process",
            "name": "Toy Process",
            "exchanges": [
                {
                    "@id": "product",
                    "amount": 1.0,
                    "flow": {"@id": "flow-product", "name": "Product"},
                    "unit": {"name": "kg"},
                    "quantitativeReference": True,
                },
                {
                    "@id": "elem-1",
                    "amount": 10.0,
                    "flow": {"@id": "flow-co2", "name": "CO2"},
                    "unit": {"name": "kg"},
                },
                {
                    "@id": "elem-2",
                    "amount": 0.1,
                    "flow": {"@id": "flow-ch4", "name": "CH4"},
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
        "flows/flow-ch4.json": {
            "@id": "flow-ch4",
            "@type": "Flow",
            "name": "CH4",
            "flowType": "ELEMENTARY_FLOW",
            "categoryPath": "air/urban air",
        },
    }
    with zipfile.ZipFile(db_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for rel_path, data in files.items():
            archive.writestr(rel_path, json.dumps(data))
    return db_zip


def test_extract_run_metadata_prefers_summary_files(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_json(
        run_dir / "run_summary.json",
        {
            "tau": 0.95,
            "input_database_zip": "/data/example_database.zip",
            "output_zip": str(run_dir / "example_lite_tau_0.95_all.zip"),
            "n_elementary_before": 100,
            "n_elementary_after": 82,
            "n_elementary_removed": 18,
            "n_coverage_failures": 0,
        },
    )

    run = extract_run_metadata(str(run_dir))

    assert run.tau == 0.95
    assert run.inputDatabaseName == "example_database.zip"
    assert run.validationStatus == "pass"
    assert run.elementaryBefore == 100
    assert run.elementaryAfter == 82
    assert run.elementaryRemoved == 18
    assert run.retainedPercent == 82.0
    assert run.removedPercent == 18.0
    assert curve_point_is_valid(run)
    assert run.warnings == []


def test_extract_run_metadata_falls_back_to_manifest(tmp_path: Path) -> None:
    run_dir = tmp_path / "manifest_only"
    run_dir.mkdir()
    _write_json(
        run_dir / "run_summary.json",
        {
            "tau": 0.8,
            "input_database_zip": "/data/manifest_database.zip",
        },
    )
    _write_manifest(
        run_dir / "process_manifest.csv",
        [
            {
                "process_id": "p1",
                "process_name": "P1",
                "n_total_exchanges_before": 0,
                "n_total_exchanges_after": 0,
                "n_elementary_before": 12,
                "n_elementary_after": 10,
                "n_elementary_characterised": 0,
                "n_elementary_uncharacterised": 0,
                "n_uncharacterised_kept": 0,
                "n_uncharacterised_removed": 0,
                "n_elementary_removed": 2,
                "n_protected_exchanges": 0,
                "positive_cover_ok": True,
                "negative_cover_ok": True,
                "coverage_failure": False,
                "status": "modified",
                "warnings": "",
            },
            {
                "process_id": "p2",
                "process_name": "P2",
                "n_total_exchanges_before": 0,
                "n_total_exchanges_after": 0,
                "n_elementary_before": 8,
                "n_elementary_after": 6,
                "n_elementary_characterised": 0,
                "n_elementary_uncharacterised": 0,
                "n_uncharacterised_kept": 0,
                "n_uncharacterised_removed": 0,
                "n_elementary_removed": 2,
                "n_protected_exchanges": 0,
                "positive_cover_ok": True,
                "negative_cover_ok": True,
                "coverage_failure": False,
                "status": "modified",
                "warnings": "",
            },
        ],
    )

    run = extract_run_metadata(str(run_dir))

    assert run.validationStatus == "pass"
    assert run.elementaryBefore == 20
    assert run.elementaryAfter == 16
    assert run.elementaryRemoved == 4
    assert run.retainedPercent == 80.0
    assert run.removedPercent == 20.0
    assert run.warnings == []


def test_extract_run_metadata_scans_output_zip_when_needed(tmp_path: Path) -> None:
    run_dir = tmp_path / "zip_only"
    run_dir.mkdir()
    output_zip = _make_toy_database_zip(run_dir, name="toy_lite_tau_0.9_all.zip")
    _write_json(
        run_dir / "run_summary.json",
        {
            "tau": 0.9,
            "input_database_zip": "/data/original_database.zip",
            "output_zip": str(output_zip),
        },
    )

    run = extract_run_metadata(str(output_zip))

    assert run.tau == 0.9
    assert run.elementaryAfter == 2
    assert run.elementaryBefore is None
    assert run.retainedPercent is None
    assert "Original elementary-exchange count is missing." in run.warnings
    assert "`run_summary.json` was not found." not in run.warnings


def test_group_warnings_detect_duplicates_and_inconsistent_counts() -> None:
    run_a = TauReductionRun(
        id="a",
        sourceFileName="run_a",
        tau=0.8,
        elementaryBefore=100,
        elementaryAfter=70,
        elementaryRemoved=30,
        retainedPercent=70.0,
        removedPercent=30.0,
        validationStatus="pass",
    )
    run_b = TauReductionRun(
        id="b",
        sourceFileName="run_b",
        tau=0.8,
        elementaryBefore=120,
        elementaryAfter=90,
        elementaryRemoved=30,
        retainedPercent=75.0,
        removedPercent=25.0,
        validationStatus="fail",
    )

    warnings = group_warnings([run_a, run_b])

    assert "Inconsistent original elementary-exchange counts within this database group." in warnings["a"]
    assert "Tau is duplicated within this database group." in warnings["a"]
    assert "Coverage validation failed for this run." in warnings["b"]
    assert "This run does not have enough metadata for curve plotting." in warnings["b"]
