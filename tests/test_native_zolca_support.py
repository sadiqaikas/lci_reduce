import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from lci_reduce.errors import NativeArchiveConversionError
from lci_reduce.native_archive import build_native_archive_from_jsonld, find_openlca_runtime


def make_toy_database_zip(base: Path, name: str = "toy_db.zip") -> Path:
    database_zip = base / name
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
                    "unit": {"@id": "unit-kg", "name": "kg"},
                    "flowProperty": {"@id": "fp-mass", "name": "Mass"},
                    "quantitativeReference": True,
                },
                {
                    "@id": "elem-1",
                    "amount": 10.0,
                    "flow": {"@id": "flow-co2", "name": "CO2"},
                    "unit": {"@id": "unit-kg", "name": "kg"},
                    "flowProperty": {"@id": "fp-mass", "name": "Mass"},
                },
                {
                    "@id": "elem-2",
                    "amount": 0.1,
                    "flow": {"@id": "flow-ch4", "name": "CH4"},
                    "unit": {"@id": "unit-kg", "name": "kg"},
                    "flowProperty": {"@id": "fp-mass", "name": "Mass"},
                },
            ],
        },
        "flows/flow-product.json": {
            "@id": "flow-product",
            "@type": "Flow",
            "name": "Product",
            "flowType": "PRODUCT_FLOW",
            "categoryPath": "products",
            "flowProperties": [
                {
                    "@type": "FlowPropertyFactor",
                    "conversionFactor": 1.0,
                    "isRefFlowProperty": True,
                    "flowProperty": {"@id": "fp-mass", "name": "Mass"},
                }
            ],
        },
        "flows/flow-co2.json": {
            "@id": "flow-co2",
            "@type": "Flow",
            "name": "CO2",
            "flowType": "ELEMENTARY_FLOW",
            "categoryPath": "air/urban air",
            "flowProperties": [
                {
                    "@type": "FlowPropertyFactor",
                    "conversionFactor": 1.0,
                    "isRefFlowProperty": True,
                    "flowProperty": {"@id": "fp-mass", "name": "Mass"},
                }
            ],
        },
        "flows/flow-ch4.json": {
            "@id": "flow-ch4",
            "@type": "Flow",
            "name": "CH4",
            "flowType": "ELEMENTARY_FLOW",
            "categoryPath": "air/urban air",
            "flowProperties": [
                {
                    "@type": "FlowPropertyFactor",
                    "conversionFactor": 1.0,
                    "isRefFlowProperty": True,
                    "flowProperty": {"@id": "fp-mass", "name": "Mass"},
                }
            ],
        },
        "flow_properties/fp-mass.json": {
            "@id": "fp-mass",
            "@type": "FlowProperty",
            "name": "Mass",
            "flowPropertyType": "PHYSICAL_QUANTITY",
            "unitGroup": {"@id": "ug-mass", "name": "Units of mass"},
        },
        "unit_groups/ug-mass.json": {
            "@id": "ug-mass",
            "@type": "UnitGroup",
            "name": "Units of mass",
            "referenceUnit": {"@id": "unit-kg", "name": "kg"},
            "units": [
                {
                    "@id": "unit-kg",
                    "name": "kg",
                    "conversionFactor": 1.0,
                    "isReferenceUnit": True,
                }
            ],
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
                {
                    "flow": {"@id": "flow-co2", "name": "CO2"},
                    "value": 10.0,
                    "unit": {"@id": "unit-kg", "name": "kg"},
                    "flowProperty": {"@id": "fp-mass", "name": "Mass"},
                },
                {
                    "flow": {"@id": "flow-ch4", "name": "CH4"},
                    "value": 0.1,
                    "unit": {"@id": "unit-kg", "name": "kg"},
                    "flowProperty": {"@id": "fp-mass", "name": "Mass"},
                },
            ],
        },
    }
    with zipfile.ZipFile(database_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for rel_path, data in files.items():
            archive.writestr(rel_path, json.dumps(data))
    return database_zip


def make_toy_database_without_methods_zip(base: Path, name: str = "toy_db_no_methods.zip") -> Path:
    database_zip = base / name
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
                    "unit": {"@id": "unit-kg", "name": "kg"},
                    "flowProperty": {"@id": "fp-mass", "name": "Mass"},
                    "quantitativeReference": True,
                },
                {
                    "@id": "elem-1",
                    "amount": 10.0,
                    "flow": {"@id": "flow-co2", "name": "CO2"},
                    "unit": {"@id": "unit-kg", "name": "kg"},
                    "flowProperty": {"@id": "fp-mass", "name": "Mass"},
                },
                {
                    "@id": "elem-2",
                    "amount": 0.1,
                    "flow": {"@id": "flow-ch4", "name": "CH4"},
                    "unit": {"@id": "unit-kg", "name": "kg"},
                    "flowProperty": {"@id": "fp-mass", "name": "Mass"},
                },
            ],
        },
        "flows/flow-product.json": {
            "@id": "flow-product",
            "@type": "Flow",
            "name": "Product",
            "flowType": "PRODUCT_FLOW",
            "categoryPath": "products",
            "flowProperties": [
                {
                    "@type": "FlowPropertyFactor",
                    "conversionFactor": 1.0,
                    "isRefFlowProperty": True,
                    "flowProperty": {"@id": "fp-mass", "name": "Mass"},
                }
            ],
        },
        "flows/flow-co2.json": {
            "@id": "flow-co2",
            "@type": "Flow",
            "name": "CO2",
            "flowType": "ELEMENTARY_FLOW",
            "categoryPath": "air/urban air",
            "flowProperties": [
                {
                    "@type": "FlowPropertyFactor",
                    "conversionFactor": 1.0,
                    "isRefFlowProperty": True,
                    "flowProperty": {"@id": "fp-mass", "name": "Mass"},
                }
            ],
        },
        "flows/flow-ch4.json": {
            "@id": "flow-ch4",
            "@type": "Flow",
            "name": "CH4",
            "flowType": "ELEMENTARY_FLOW",
            "categoryPath": "air/urban air",
            "flowProperties": [
                {
                    "@type": "FlowPropertyFactor",
                    "conversionFactor": 1.0,
                    "isRefFlowProperty": True,
                    "flowProperty": {"@id": "fp-mass", "name": "Mass"},
                }
            ],
        },
        "flow_properties/fp-mass.json": {
            "@id": "fp-mass",
            "@type": "FlowProperty",
            "name": "Mass",
            "flowPropertyType": "PHYSICAL_QUANTITY",
            "unitGroup": {"@id": "ug-mass", "name": "Units of mass"},
        },
        "unit_groups/ug-mass.json": {
            "@id": "ug-mass",
            "@type": "UnitGroup",
            "name": "Units of mass",
            "referenceUnit": {"@id": "unit-kg", "name": "kg"},
            "units": [
                {
                    "@id": "unit-kg",
                    "name": "kg",
                    "conversionFactor": 1.0,
                    "isReferenceUnit": True,
                }
            ],
        },
    }
    with zipfile.ZipFile(database_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for rel_path, data in files.items():
            archive.writestr(rel_path, json.dumps(data))
    return database_zip


def make_methods_only_zip(base: Path, name: str = "methods_only.zip") -> Path:
    methods_zip = base / name
    with zipfile.ZipFile(methods_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "flow_properties/fp-mass.json",
            json.dumps(
                {
                    "@id": "fp-mass",
                    "@type": "FlowProperty",
                    "name": "Mass",
                    "flowPropertyType": "PHYSICAL_QUANTITY",
                    "unitGroup": {"@id": "ug-mass", "name": "Units of mass"},
                }
            ),
        )
        archive.writestr(
            "unit_groups/ug-mass.json",
            json.dumps(
                {
                    "@id": "ug-mass",
                    "@type": "UnitGroup",
                    "name": "Units of mass",
                    "referenceUnit": {"@id": "unit-kg", "name": "kg"},
                    "units": [
                        {
                            "@id": "unit-kg",
                            "name": "kg",
                            "conversionFactor": 1.0,
                            "isReferenceUnit": True,
                        }
                    ],
                }
            ),
        )
        archive.writestr(
            "lcia_methods/method-1.json",
            json.dumps(
                {
                    "@id": "method-1",
                    "@type": "ImpactMethod",
                    "name": "IPCC 2021",
                }
            ),
        )
        archive.writestr(
            "lcia_categories/category-1.json",
            json.dumps(
                {
                    "@id": "category-1",
                    "@type": "ImpactCategory",
                    "name": "Climate change",
                    "impactMethod": {"@id": "method-1", "name": "IPCC 2021"},
                    "referenceUnitName": "kg",
                    "impactFactors": [
                        {
                            "flow": {"@id": "flow-co2", "name": "CO2"},
                            "value": 10.0,
                            "unit": {"@id": "unit-kg", "name": "kg"},
                            "flowProperty": {"@id": "fp-mass", "name": "Mass"},
                        },
                        {
                            "flow": {"@id": "flow-ch4", "name": "CH4"},
                            "value": 0.1,
                            "unit": {"@id": "unit-kg", "name": "kg"},
                            "flowProperty": {"@id": "fp-mass", "name": "Mass"},
                        },
                    ],
                }
            ),
        )
    return methods_zip


def _require_native_support():
    try:
        find_openlca_runtime()
        return True
    except NativeArchiveConversionError:
        return False


native_support = pytest.mark.skipif(
    not _require_native_support(),
    reason="Native openLCA support requires a local openLCA installation",
)


@native_support
def test_native_zolca_database_with_embedded_methods(tmp_path: Path):
    jsonld_zip = make_toy_database_zip(tmp_path, name="toy_db.zip")
    native_zolca = tmp_path / "toy_db_native.zolca"
    build_native_archive_from_jsonld(str(jsonld_zip), str(native_zolca))

    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd())

    inspect_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "lci_reduce.cli",
            "inspect",
            "--database",
            str(native_zolca),
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

    output_dir = tmp_path / "out_native_embedded"
    create_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "lci_reduce.cli",
            "create",
            "--database",
            str(native_zolca),
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


@native_support
def test_native_zolca_database_and_native_methods_archive(tmp_path: Path):
    database_jsonld = make_toy_database_without_methods_zip(tmp_path)
    methods_jsonld = make_methods_only_zip(tmp_path)
    native_db = tmp_path / "toy_db_no_methods_native.zolca"
    native_methods = tmp_path / "methods_native.zolca"
    build_native_archive_from_jsonld(str(database_jsonld), str(native_db))
    build_native_archive_from_jsonld(str(methods_jsonld), str(native_methods))

    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd())
    output_dir = tmp_path / "out_native_methods"
    create_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "lci_reduce.cli",
            "create",
            "--database",
            str(native_db),
            "--methods",
            str(native_methods),
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
