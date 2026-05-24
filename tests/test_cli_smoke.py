import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

from lci_reduce.cli import create_command, inspect_command


def make_toy_database_zip(base: Path, name: str = "toy_db.zip") -> Path:
    import zipfile

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
                {"flow": {"@id": "flow-co2", "name": "CO2"}, "value": 10.0, "unitName": "kg"},
                {"flow": {"@id": "flow-ch4", "name": "CH4"}, "value": 0.1, "unitName": "kg"},
            ],
        },
    }
    with zipfile.ZipFile(db_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for rel_path, data in files.items():
            archive.writestr(rel_path, json.dumps(data))
    return db_zip


def make_external_methods_zip(base: Path, name: str = "external_methods.zip") -> Path:
    methods_zip = base / name
    files = {
        "external_methods/method-ext.json": {
            "@id": "method-ext",
            "@type": "ImpactMethod",
            "name": "External IPCC 2024",
        },
        "external_categories/category-1.json": {
            "@id": "category-1",
            "@type": "ImpactCategory",
            "name": "Climate change",
            "impactMethod": {"@id": "method-ext", "name": "External IPCC 2024"},
            "referenceUnitName": "kg",
            "impactFactors": [
                {"flow": {"@id": "flow-co2", "name": "CO2"}, "value": 42.0, "unitName": "kg"},
                {"flow": {"@id": "flow-ch4", "name": "CH4"}, "value": 2.5, "unitName": "kg"},
            ],
        },
    }
    with zipfile.ZipFile(methods_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for rel_path, data in files.items():
            archive.writestr(rel_path, json.dumps(data))
    return methods_zip


def make_ambiguous_cf_database_zip(base: Path, name: str = "ambiguous_db.zip") -> Path:
    db_zip = base / name
    files = {
        "processes/process-1.json": {
            "@id": "process-1",
            "@type": "Process",
            "name": "Ambiguous Process",
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
                    "amount": 1.0,
                    "flow": {"@id": "flow-methane", "name": "Methane, biogenic"},
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
        "flows/flow-methane.json": {
            "@id": "flow-methane",
            "@type": "Flow",
            "name": "Methane, biogenic",
            "flowType": "ELEMENTARY_FLOW",
            "categoryPath": "air/urban air",
        },
        "lcia_methods/method-1.json": {
            "@id": "method-1",
            "@type": "ImpactMethod",
            "name": "Regional Method",
        },
        "lcia_categories/category-1.json": {
            "@id": "category-1",
            "@type": "ImpactCategory",
            "name": "Ecotoxicity, freshwater",
            "impactMethod": {"@id": "method-1", "name": "Regional Method"},
            "referenceUnitName": "kg",
            "impactFactors": [
                {"flow": {"@id": "flow-methane", "name": "Methane, biogenic"}, "value": 1.0, "unitName": "kg"},
                {"flow": {"@id": "flow-methane", "name": "Methane, biogenic"}, "value": 2.0, "unitName": "kg"},
            ],
        },
    }
    with zipfile.ZipFile(db_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for rel_path, data in files.items():
            archive.writestr(rel_path, json.dumps(data))
    return db_zip


def make_method_reference_database_zip(base: Path, name: str = "method_refs_db.zip") -> Path:
    db_zip = base / name
    files = {
        "processes/process-1.json": {
            "@id": "process-1",
            "@type": "Process",
            "name": "Method Ref Process",
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
                    "amount": 2.0,
                    "flow": {"@id": "flow-co2", "name": "CO2"},
                    "unit": {"name": "kg"},
                },
                {
                    "@id": "elem-2",
                    "amount": 3.0,
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
        "lcia_methods/method-1.json": {
            "@id": "method-1",
            "@type": "ImpactMethod",
            "name": "Recovered Method",
            "categoryPath": "Regulatory/Climate",
            "impactCategories": [
                {"@id": "category-1", "name": "Climate change"},
                {"@id": "category-2", "name": "Fossil methane"},
            ],
        },
        "lcia_categories/category-1.json": {
            "@id": "category-1",
            "@type": "ImpactCategory",
            "name": "Climate change",
            "referenceUnitName": "kg",
            "impactFactors": [
                {"flow": {"@id": "flow-co2", "name": "CO2"}, "value": 10.0, "unitName": "kg"},
            ],
        },
        "lcia_categories/category-2.json": {
            "@id": "category-2",
            "@type": "ImpactCategory",
            "name": "Fossil methane",
            "referenceUnitName": "kg",
            "impactFactors": [
                {"flow": {"@id": "flow-ch4", "name": "CH4"}, "value": 1.5, "unitName": "kg"},
            ],
        },
    }
    with zipfile.ZipFile(db_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for rel_path, data in files.items():
            archive.writestr(rel_path, json.dumps(data))
    return db_zip


def make_empty_category_database_zip(base: Path, name: str = "empty_category_db.zip") -> Path:
    db_zip = base / name
    files = {
        "processes/process-1.json": {
            "@id": "process-1",
            "@type": "Process",
            "name": "Empty Category Process",
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
                    "amount": 2.0,
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
            "categoryPath": "products",
        },
        "flows/flow-co2.json": {
            "@id": "flow-co2",
            "@type": "Flow",
            "name": "CO2",
            "flowType": "ELEMENTARY_FLOW",
            "categoryPath": "air/urban air",
        },
        "lcia_methods/method-1.json": {
            "@id": "method-1",
            "@type": "ImpactMethod",
            "name": "Method With Empty Category",
            "categoryPath": "Methods/Diagnostics",
        },
        "lcia_categories/category-empty.json": {
            "@id": "category-empty",
            "@type": "ImpactCategory",
            "name": "Empty category",
            "impactMethod": {"@id": "method-1", "name": "Method With Empty Category"},
            "referenceUnitName": "kg",
            "impactFactors": [],
        },
    }
    with zipfile.ZipFile(db_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for rel_path, data in files.items():
            archive.writestr(rel_path, json.dumps(data))
    return db_zip


def test_cli_smoke(tmp_path: Path):
    db_zip = make_toy_database_zip(tmp_path)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd())

    inspect_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "lci_reduce.cli",
            "inspect",
            "--database",
            str(db_zip),
        ],
        cwd=Path.cwd(),
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    inspect_json = json.loads(inspect_result.stdout)
    assert inspect_json["detected_processes"] == 1
    assert inspect_json["database_contains_impact_methods"] is True
    assert inspect_json["database_lcia_methods"] == 1
    assert inspect_json["database_lcia_categories"] == 1
    assert "leave --methods empty" in inspect_json["database_methods_hint"]
    assert inspect_json["lcia_categories"][0]["method_name"] == "IPCC 2021"
    assert inspect_json["lcia_categories"][0]["source_file"] == "lcia_categories/category-1.json"

    output_dir = tmp_path / "out"
    create_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "lci_reduce.cli",
            "create",
            "--database",
            str(db_zip),
            "--output",
            str(output_dir),
            "--tau",
            "0.95",
            "--method-selection",
            "all",
            "--uncharacterised-policy",
            "keep",
            "--strict-units",
            "true",
        ],
        cwd=Path.cwd(),
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    result_json = json.loads(create_result.stdout)
    assert Path(result_json["output_zip"]).exists()
    assert Path(result_json["exchange_manifest_csv"]).exists()
    assert Path(result_json["process_manifest_csv"]).exists()
    assert Path(result_json["run_summary_json"]).exists()


def test_cli_smoke_zolca_extension(tmp_path: Path):
    db_zolca = make_toy_database_zip(tmp_path, name="toy_db.zolca")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd())

    inspect_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "lci_reduce.cli",
            "inspect",
            "--database",
            str(db_zolca),
        ],
        cwd=Path.cwd(),
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    inspect_json = json.loads(inspect_result.stdout)
    assert inspect_json["detected_processes"] == 1
    assert inspect_json["database_contains_impact_methods"] is True

    output_dir = tmp_path / "out_zolca"
    create_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "lci_reduce.cli",
            "create",
            "--database",
            str(db_zolca),
            "--output",
            str(output_dir),
            "--tau",
            "0.95",
            "--method-selection",
            "all",
            "--uncharacterised-policy",
            "keep",
            "--strict-units",
            "true",
        ],
        cwd=Path.cwd(),
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    result_json = json.loads(create_result.stdout)
    assert Path(result_json["output_zip"]).exists()


def test_cli_conflicting_cfs_fail_and_write_diagnostics(tmp_path: Path):
    db_zip = make_ambiguous_cf_database_zip(tmp_path)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd())
    output_dir = tmp_path / "out_ambiguous"
    choices_path = tmp_path / "choices.csv"

    create_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "lci_reduce.cli",
            "create",
            "--database",
            str(db_zip),
            "--output",
            str(output_dir),
            "--tau",
            "0.95",
            "--method-selection",
            "all",
            "--uncharacterised-policy",
            "keep",
            "--strict-units",
            "true",
            "--cf-resolution-file",
            str(choices_path),
        ],
        cwd=Path.cwd(),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert create_result.returncode != 0
    assert "Multiple conflicting characterisation factor candidates remained after disambiguation" in create_result.stderr
    run_summary = json.loads((output_dir / "run_summary.json").read_text(encoding="utf-8"))
    assert [record["cf_value"] for record in run_summary["cf_ambiguities"]] == ["1", "2"]
    assert choices_path.exists()


def test_create_progress_callback_reports_steps(tmp_path: Path):
    db_zip = make_toy_database_zip(tmp_path)
    output_dir = tmp_path / "progress_out"
    updates = []

    result = create_command(
        database=str(db_zip),
        methods=None,
        output=str(output_dir),
        tau=0.95,
        method_selection="all",
        uncharacterised_policy="keep",
        strict_units=True,
        tolerance=1e-12,
        progress_callback=updates.append,
    )

    assert Path(result.output_zip).exists()
    assert updates
    steps = [update.step for update in updates]
    assert "load_database" in steps
    assert "reduce_processes" in steps
    assert "write_summary" in steps
    assert updates[-1].message == "Step 8/8: Run completed."


def test_inspect_uses_external_methods_when_provided(tmp_path: Path):
    db_zip = make_toy_database_zip(tmp_path)
    methods_zip = make_external_methods_zip(tmp_path)

    inspect_json = inspect_command(str(db_zip), str(methods_zip))

    assert inspect_json["database_contains_impact_methods"] is True
    assert inspect_json["database_lcia_methods"] == 1
    assert inspect_json["database_lcia_categories"] == 1
    assert inspect_json["detected_lcia_methods"] == 1
    assert inspect_json["detected_lcia_categories"] == 1
    assert inspect_json["lcia_categories"][0]["method_id"] == "method-ext"
    assert inspect_json["lcia_categories"][0]["method_name"] == "External IPCC 2024"
    assert inspect_json["lcia_categories"][0]["source_file"] == "external_categories/category-1.json"


def test_create_uses_external_methods_when_provided_and_reports_summary_source(tmp_path: Path):
    db_zip = make_toy_database_zip(tmp_path)
    methods_zip = make_external_methods_zip(tmp_path)
    output_dir = tmp_path / "external_methods_out"

    result = create_command(
        database=str(db_zip),
        methods=str(methods_zip),
        output=str(output_dir),
        tau=0.95,
        method_selection="all",
        uncharacterised_policy="keep",
        strict_units=True,
        tolerance=1e-12,
    )

    run_summary = json.loads(Path(result.run_summary_json).read_text(encoding="utf-8"))

    assert run_summary["lcia_method_source"] == "external"
    assert run_summary["internal_lcia_methods_ignored"] is True
    assert {row["method_id"] for row in run_summary["selected_lcia_categories"]} == {"method-ext"}
    assert {row["method_name"] for row in run_summary["selected_lcia_categories"]} == {"External IPCC 2024"}
    assert {row["source_file"] for row in run_summary["selected_lcia_categories"]} == {
        "external_categories/category-1.json"
    }
    with zipfile.ZipFile(result.output_zip, "r") as archive:
        assert "external_categories/category-1.json" not in archive.namelist()
        assert "external_methods/method-ext.json" not in archive.namelist()


def test_method_file_category_references_recover_parent_method_reporting(tmp_path: Path):
    db_zip = make_method_reference_database_zip(tmp_path)
    output_dir = tmp_path / "method_refs_out"
    inspect_json = inspect_command(str(db_zip), None)

    result = create_command(
        database=str(db_zip),
        methods=None,
        output=str(output_dir),
        tau=0.95,
        method_selection="all",
        uncharacterised_policy="keep",
        strict_units=True,
        tolerance=1e-12,
    )

    run_summary = json.loads(Path(result.run_summary_json).read_text(encoding="utf-8"))

    selected_rows = run_summary["selected_lcia_categories"]
    assert run_summary["lcia_method_source"] == "database"
    assert run_summary["internal_lcia_methods_ignored"] is False
    assert len(selected_rows) == 2
    assert {row["method_name"] for row in inspect_json["lcia_categories"]} == {"Recovered Method"}
    assert {row["method_name"] for row in selected_rows} == {"Recovered Method"}
    assert {row["method_id"] for row in selected_rows} == {"method-1"}
    assert {row["method_path"] for row in selected_rows} == {"Regulatory/Climate"}
    assert {row["source_file"] for row in selected_rows} == {
        "lcia_categories/category-1.json",
        "lcia_categories/category-2.json",
    }
    assert run_summary["selected_lcia_categories"][0]["method_name"] == "Recovered Method"


def test_empty_lcia_category_reporting_is_detailed(tmp_path: Path):
    db_zip = make_empty_category_database_zip(tmp_path)
    output_dir = tmp_path / "empty_category_out"

    result = create_command(
        database=str(db_zip),
        methods=None,
        output=str(output_dir),
        tau=0.95,
        method_selection="all",
        uncharacterised_policy="keep",
        strict_units=True,
        tolerance=1e-12,
    )

    run_summary = json.loads(Path(result.run_summary_json).read_text(encoding="utf-8"))
    warnings_text = json.dumps(run_summary["warnings"], ensure_ascii=True)

    assert run_summary["n_empty_lcia_categories"] == 1
    assert run_summary["empty_lcia_categories"][0]["category_id"] == "category-empty"
    assert run_summary["empty_lcia_categories"][0]["method_name"] == "Method With Empty Category"
    assert "category-empty" in warnings_text
    assert "Method With Empty Category" in warnings_text
    assert "lcia_categories/category-empty.json" in warnings_text


def test_native_zolca_without_json_fails_clearly(tmp_path: Path):
    import zipfile

    bad_zolca = tmp_path / "native_db.zolca"
    with zipfile.ZipFile(bad_zolca, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("database.properties", "version=2")
        archive.writestr("database.script", "not jsonld")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd())
    inspect_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "lci_reduce.cli",
            "inspect",
            "--database",
            str(bad_zolca),
        ],
        cwd=Path.cwd(),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert inspect_result.returncode != 0
    assert "No process objects found" in inspect_result.stderr
