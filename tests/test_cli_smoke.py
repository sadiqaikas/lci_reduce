import csv
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

from lci_reduce import priority_glad as priority_glad_module
from lci_reduce.cli import create_command, inspect_command, main, priority_command
from lci_reduce.errors import MissingFlowError
from lci_reduce.archive_reader import index_archive, parse_json_object
from lci_reduce.lcia import select_lcia_categories
from lci_reduce.reducer import reduce_process


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


def make_fail_fast_database_zip(base: Path, name: str = "fail_fast_db.zip") -> Path:
    db_zip = base / name
    files = {
        "processes/process-0-bad.json": {
            "@id": "process-bad",
            "@type": "Process",
            "name": "Bad Process",
            "exchanges": [
                {
                    "@id": "elem-missing",
                    "amount": 1.0,
                    "flow": {"@id": "flow-missing", "name": "Missing"},
                    "unit": {"name": "kg"},
                }
            ],
        },
        "processes/process-1-good.json": {
            "@id": "process-good",
            "@type": "Process",
            "name": "Good Process",
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


def _read_zip_json(zip_path: Path, rel_path: str) -> dict:
    with zipfile.ZipFile(zip_path, "r") as archive:
        return json.loads(archive.read(rel_path).decode("utf-8"))


def _read_debug_lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


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


def make_external_methods_zip_with_gram_units(base: Path, name: str = "external_methods_g.zip") -> Path:
    methods_zip = base / name
    files = {
        "external_methods/method-ext.json": {
            "@id": "method-ext",
            "@type": "ImpactMethod",
            "name": "External Mass Method",
        },
        "external_categories/category-1.json": {
            "@id": "category-1",
            "@type": "ImpactCategory",
            "name": "Climate change",
            "impactMethod": {"@id": "method-ext", "name": "External Mass Method"},
            "referenceUnitName": "kg",
            "impactFactors": [
                {
                    "flow": {"@id": "flow-co2", "name": "CO2"},
                    "value": 0.042,
                    "unit": {"@id": "unit-g", "name": "g"},
                },
                {
                    "flow": {"@id": "flow-ch4", "name": "CH4"},
                    "value": 0.0025,
                    "unit": {"@id": "unit-g", "name": "g"},
                },
            ],
        },
        "unit_groups/ug-mass.json": {
            "@id": "ug-mass",
            "@type": "UnitGroup",
            "name": "Units of mass",
            "referenceUnit": {"@id": "unit-kg", "name": "kg"},
            "units": [
                {
                    "@id": "unit-g",
                    "name": "g",
                    "conversionFactor": 0.001,
                }
            ],
        },
    }
    with zipfile.ZipFile(methods_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for rel_path, data in files.items():
            archive.writestr(rel_path, json.dumps(data))
    return methods_zip


def _write_priority_match_csv(path: Path) -> Path:
    fieldnames = [
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
        "loss_max_0_95",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "flow_id": "flow-a",
                "flow_name": "Alpha",
                "compartment": "air",
                "subcompartment": "urban air",
                "reference_unit": "kg",
                "occurrence_count": "1",
                "characterised_occurrence_count": "1",
                "tau_entry_min": "0.1",
                "tau_entry_median": "0.1",
                "tau_entry_max": "0.1",
                "eta_0_95": "0",
                "loss_max_0_95": "0",
            }
        )
    return path


def _write_glad_asset_dir(path: Path) -> Path:
    fieldnames = [
        "source_file",
        "list_name",
        "flow_uuid",
        "flow_name",
        "context",
        "unit",
        "cas_number",
        "synonyms",
    ]
    row = {
        "source_file": "snapshot.csv",
        "list_name": "target",
        "flow_uuid": "flow-a",
        "flow_name": "Alpha",
        "context": "air/urban air",
        "unit": "kg",
        "cas_number": "",
        "synonyms": "",
    }
    path.mkdir(parents=True, exist_ok=True)
    manifest = {"targets": {}, "upstream_repo_url": "https://example.invalid/glad"}
    for name in ("ecoinventEFv3.7", "ILCD_EFv3.0", "FEDEFLv1.0.3", "IDEA_EFv2.3"):
        snapshot = path / f"{name}.csv"
        with snapshot.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow({**row, "list_name": name, "source_file": snapshot.name})
        manifest["targets"][name] = {"snapshot": snapshot.name}
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return path


def make_external_methods_zip_with_gram_names_only(base: Path, name: str = "external_methods_g_names_only.zip") -> Path:
    methods_zip = base / name
    files = {
        "external_methods/method-ext.json": {
            "@id": "method-ext",
            "@type": "ImpactMethod",
            "name": "External Mass Method Names Only",
        },
        "external_categories/category-1.json": {
            "@id": "category-1",
            "@type": "ImpactCategory",
            "name": "Climate change",
            "impactMethod": {"@id": "method-ext", "name": "External Mass Method Names Only"},
            "referenceUnitName": "kg",
            "impactFactors": [
                {
                    "flow": {"@id": "flow-co2", "name": "CO2"},
                    "value": 0.042,
                    "unitName": "g",
                },
                {
                    "flow": {"@id": "flow-ch4", "name": "CH4"},
                    "value": 0.0025,
                    "unitName": "g",
                },
            ],
        },
    }
    with zipfile.ZipFile(methods_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for rel_path, data in files.items():
            archive.writestr(rel_path, json.dumps(data))
    return methods_zip


def make_external_methods_zip_with_becquerel_names_only(
    base: Path,
    name: str = "external_methods_bq_names_only.zip",
) -> Path:
    methods_zip = base / name
    files = {
        "external_methods/method-ext.json": {
            "@id": "method-ext",
            "@type": "ImpactMethod",
            "name": "External Activity Method Names Only",
        },
        "external_categories/category-1.json": {
            "@id": "category-1",
            "@type": "ImpactCategory",
            "name": "Ionizing radiation",
            "impactMethod": {"@id": "method-ext", "name": "External Activity Method Names Only"},
            "referenceUnitName": "kBq",
            "impactFactors": [
                {
                    "flow": {"@id": "flow-u235", "name": "Uranium-235"},
                    "value": 0.5,
                    "unitName": "Bq",
                }
            ],
        },
    }
    with zipfile.ZipFile(methods_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for rel_path, data in files.items():
            archive.writestr(rel_path, json.dumps(data))
    return methods_zip


def make_database_with_support_objects_zip(base: Path, name: str = "support_db.zip") -> Path:
    db_zip = base / name
    files = {
        "processes/process-1.json": {
            "@id": "process-1",
            "@type": "Process",
            "name": "Support Object Process",
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
    with zipfile.ZipFile(db_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for rel_path, data in files.items():
            archive.writestr(rel_path, json.dumps(data))
    return db_zip


def make_database_with_activity_support_objects_zip(base: Path, name: str = "support_activity_db.zip") -> Path:
    db_zip = base / name
    files = {
        "processes/process-1.json": {
            "@id": "process-1",
            "@type": "Process",
            "name": "Support Activity Process",
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
                    "amount": 2.0,
                    "flow": {"@id": "flow-u235", "name": "Uranium-235"},
                    "unit": {"@id": "unit-kbq", "name": "kBq"},
                    "flowProperty": {"@id": "fp-activity", "name": "Activity"},
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
        "flows/flow-u235.json": {
            "@id": "flow-u235",
            "@type": "Flow",
            "name": "Uranium-235",
            "flowType": "ELEMENTARY_FLOW",
            "categoryPath": "resource/in ground",
            "flowProperties": [
                {
                    "@type": "FlowPropertyFactor",
                    "conversionFactor": 1.0,
                    "isRefFlowProperty": True,
                    "flowProperty": {"@id": "fp-activity", "name": "Activity"},
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
        "flow_properties/fp-activity.json": {
            "@id": "fp-activity",
            "@type": "FlowProperty",
            "name": "Activity",
            "flowPropertyType": "PHYSICAL_QUANTITY",
            "unitGroup": {"@id": "ug-activity", "name": "Units of radioactivity"},
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
        "unit_groups/ug-activity.json": {
            "@id": "ug-activity",
            "@type": "UnitGroup",
            "name": "Units of radioactivity",
            "referenceUnit": {"@id": "unit-bq", "name": "Bq"},
            "units": [
                {
                    "@id": "unit-bq",
                    "name": "Bq",
                    "conversionFactor": 1.0,
                    "isReferenceUnit": True,
                },
                {
                    "@id": "unit-kbq",
                    "name": "kBq",
                    "conversionFactor": 1000.0,
                    "isReferenceUnit": False,
                },
            ],
        },
    }
    with zipfile.ZipFile(db_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for rel_path, data in files.items():
            archive.writestr(rel_path, json.dumps(data))
    return db_zip


def make_database_with_sidecar_process_paths_zip(
    base: Path,
    name: str = "sidecar_process_db.zip",
    *,
    root_prefix: str = "",
) -> Path:
    db_zip = base / name
    files = {
        "processes/process-1.json": {
            "@id": "process-1",
            "@type": "Process",
            "name": "Real Process",
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
                    "amount": 1.0,
                    "flow": {"@id": "flow-co2", "name": "CO2"},
                    "unit": {"@id": "unit-kg", "name": "kg"},
                    "flowProperty": {"@id": "fp-mass", "name": "Mass"},
                },
            ],
        },
        "bin/processes/process-1/calculation-preferences.json": {
            "@id": "process-sidecar",
            "@type": "Process",
            "name": "Calculation preferences",
            "solver": "deterministic",
            "providerLinking": {"mode": "prefer"},
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
            "referenceUnitName": "kg CO2e",
            "impactFactors": [
                {
                    "flow": {"@id": "flow-co2", "name": "CO2"},
                    "value": 1.0,
                    "unit": {"@id": "unit-kg", "name": "kg"},
                    "flowProperty": {"@id": "fp-mass", "name": "Mass"},
                }
            ],
        },
    }
    with zipfile.ZipFile(db_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for rel_path, data in files.items():
            target_path = f"{root_prefix.rstrip('/')}/{rel_path}" if root_prefix else rel_path
            archive.writestr(target_path, json.dumps(data))
    return db_zip


def make_methods_with_duplicate_support_objects_zip(
    base: Path,
    name: str = "methods_with_duplicate_support.zip",
) -> Path:
    methods_zip = base / name
    files = {
        "flow_properties/fp-mass.json": {
            "@id": "fp-mass",
            "@type": "FlowProperty",
            "name": "Mass",
            "description": "External methods copy",
            "flowPropertyType": "PHYSICAL_QUANTITY",
            "unitGroup": {"@id": "ug-mass", "name": "Units of mass"},
        },
        "unit_groups/ug-mass.json": {
            "@id": "ug-mass",
            "@type": "UnitGroup",
            "description": "External methods copy",
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
    assert Path(result_json["run_summary_json"]).exists()
    assert Path(result_json["reduction_debug_ndjson"]).exists()


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
    assert Path(result_json["reduction_debug_ndjson"]).exists()


def test_cli_conflicting_cfs_create_succeeds_with_finite_scenario_rows(tmp_path: Path):
    db_zip = make_ambiguous_cf_database_zip(tmp_path)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd())
    output_dir = tmp_path / "out_ambiguous"

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
        check=False,
    )

    assert create_result.returncode == 0
    result_json = json.loads(create_result.stdout)
    assert Path(result_json["output_zip"]).exists()
    run_summary = json.loads((output_dir / "run_summary.json").read_text(encoding="utf-8"))
    assert run_summary["cf_resolution_policy"] == "finite_scenario_rows"
    assert run_summary["n_finite_cf_candidate_sets"] == 1
    assert run_summary["n_scenario_rows_added"] == 2
    assert run_summary["n_cf_ambiguities_unresolved"] == 0
    assert len(_read_debug_lines(output_dir / "reduction_debug.ndjson")) == 1


def test_explore_ambiguities_writes_dedicated_outputs_and_keeps_scanning(tmp_path: Path) -> None:
    db_zip = tmp_path / "ambiguity_scan.zip"
    files = {
        "processes/process-bad.json": {
            "@id": "process-bad",
            "@type": "Process",
            "name": "Broken process",
            "exchanges": [
                {
                    "@id": "product",
                    "amount": 1.0,
                    "flow": {"@id": "flow-product", "name": "Product"},
                    "unit": {"name": "kg"},
                    "quantitativeReference": True,
                },
                {
                    "@id": "elem-missing",
                    "amount": 1.0,
                    "flow": {"@id": "flow-missing", "name": "Missing flow"},
                    "unit": {"name": "kg"},
                },
            ],
        },
        "processes/process-good.json": {
            "@id": "process-good",
            "@type": "Process",
            "name": "Ambiguous process",
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

    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd())
    output_dir = tmp_path / "ambiguity_out"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "lci_reduce.cli",
            "explore-ambiguities",
            "--database",
            str(db_zip),
            "--output",
            str(output_dir),
            "--method-selection",
            "all",
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
    assert Path(payload["cf_ambiguities_csv"]).exists()
    assert Path(payload["cf_ambiguity_metadata_json"]).exists()
    assert {path.name for path in output_dir.iterdir()} == {
        "cf_ambiguities.csv",
        "cf_ambiguities_metadata.json",
    }
    metadata = json.loads(Path(payload["cf_ambiguity_metadata_json"]).read_text(encoding="utf-8"))
    assert metadata["scan_mode"] == "record_only_first_n_processes"
    assert metadata["process_scan_limit"] == 2
    assert metadata["csv_row_mode"] == "group_summary"
    assert metadata["n_processes_scanned"] == 2
    assert metadata["n_processes_failed"] == 1
    assert metadata["n_cf_ambiguity_records"] > 0
    assert metadata["group_summaries"]
    header, rows = _read_csv_rows(Path(payload["cf_ambiguities_csv"]))
    assert "candidate_flow_source_files" in header
    assert "all_candidate_metadata" not in header
    assert "candidate_rows" in header
    assert rows
    assert rows[0]["is_non_location_ambiguity"] == "true"


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
    assert Path(result.reduction_debug_ndjson).exists()
    assert updates
    steps = [update.step for update in updates]
    assert "load_database" in steps
    assert "reduce_processes" in steps
    assert "write_summary" in steps
    assert updates[-1].message == "Step 8/8: Run completed."


def test_sidecar_process_paths_do_not_inflate_counts_or_get_rewritten(tmp_path: Path):
    db_zip = make_database_with_sidecar_process_paths_zip(tmp_path)
    output_dir = tmp_path / "sidecar_process_out"
    inspect_json = inspect_command(str(db_zip), None)
    create_updates = []
    priority_updates = []

    create_result = create_command(
        database=str(db_zip),
        methods=None,
        output=str(output_dir),
        tau=0.95,
        method_selection="all",
        uncharacterised_policy="keep",
        strict_units=True,
        tolerance=1e-12,
        progress_callback=create_updates.append,
    )
    priority_result = priority_command(
        database=str(db_zip),
        methods=None,
        output=str(tmp_path / "sidecar_process_priority"),
        method_selection="all",
        audit_tau=[0.95],
        strict_units=True,
        tolerance=1e-12,
        progress_callback=priority_updates.append,
    )

    sidecar_path = "bin/processes/process-1/calculation-preferences.json"
    flow_property_path = "flow_properties/fp-mass.json"
    run_summary = json.loads(Path(create_result.run_summary_json).read_text(encoding="utf-8"))
    debug_lines = _read_debug_lines(Path(create_result.reduction_debug_ndjson))
    priority_metadata = json.loads(Path(priority_result.flow_priority_metadata_json).read_text(encoding="utf-8"))

    assert inspect_json["detected_processes"] == 1
    assert inspect_json["detected_flows"] == 2
    assert run_summary["n_processes_total"] == 1
    assert len(debug_lines) == 1
    assert priority_metadata["n_processes_total"] == 1
    assert any(update.message == "Step 1/8: Indexed 1 processes and 2 flows." for update in create_updates)
    assert any(update.message == "Step 5/8: Reducing and writing 1 processes..." for update in create_updates)
    assert any(update.message == "Step 1/6: Indexed 1 processes and 2 flows." for update in priority_updates)

    with zipfile.ZipFile(db_zip, "r") as source_archive, zipfile.ZipFile(create_result.output_zip, "r") as output_archive:
        assert json.loads(output_archive.read(sidecar_path).decode("utf-8")) == json.loads(source_archive.read(sidecar_path).decode("utf-8"))
        assert json.loads(output_archive.read(flow_property_path).decode("utf-8")) == json.loads(source_archive.read(flow_property_path).decode("utf-8"))
        assert "exchanges" not in json.loads(output_archive.read(sidecar_path).decode("utf-8"))


def test_root_prefixed_sidecar_process_paths_do_not_inflate_counts_or_get_rewritten(tmp_path: Path):
    root_prefix = "BAFU_export"
    db_zip = make_database_with_sidecar_process_paths_zip(tmp_path, name="sidecar_process_rooted_db.zip", root_prefix=root_prefix)
    output_dir = tmp_path / "sidecar_process_rooted_out"
    inspect_json = inspect_command(str(db_zip), None)
    create_updates = []

    create_result = create_command(
        database=str(db_zip),
        methods=None,
        output=str(output_dir),
        tau=0.95,
        method_selection="all",
        uncharacterised_policy="keep",
        strict_units=True,
        tolerance=1e-12,
        progress_callback=create_updates.append,
    )

    sidecar_path = f"{root_prefix}/bin/processes/process-1/calculation-preferences.json"
    process_path = f"{root_prefix}/processes/process-1.json"
    run_summary = json.loads(Path(create_result.run_summary_json).read_text(encoding="utf-8"))
    debug_lines = _read_debug_lines(Path(create_result.reduction_debug_ndjson))

    assert inspect_json["detected_processes"] == 1
    assert inspect_json["detected_flows"] == 2
    assert run_summary["n_processes_total"] == 1
    assert len(debug_lines) == 1
    assert debug_lines[0]["process_name"] == "Real Process"
    assert any(update.message == "Step 1/8: Indexed 1 processes and 2 flows." for update in create_updates)
    assert not any("calculation-preferences.json" in update.message for update in create_updates)

    with zipfile.ZipFile(db_zip, "r") as source_archive, zipfile.ZipFile(create_result.output_zip, "r") as output_archive:
        assert json.loads(output_archive.read(sidecar_path).decode("utf-8")) == json.loads(source_archive.read(sidecar_path).decode("utf-8"))
        assert [exchange["@id"] for exchange in json.loads(output_archive.read(process_path).decode("utf-8"))["exchanges"]] == ["product", "elem-1"]


def test_water_override_flag_is_recorded_in_create_and_priority_metadata(tmp_path: Path):
    db_zip = make_toy_database_zip(tmp_path, name="water_override_metadata_db.zip")

    create_result = create_command(
        database=str(db_zip),
        methods=None,
        output=str(tmp_path / "create_out"),
        tau=0.95,
        method_selection="all",
        uncharacterised_policy="keep",
        strict_units=True,
        tolerance=1e-12,
        allow_water_mass_volume_override=True,
    )
    priority_result = priority_command(
        database=str(db_zip),
        methods=None,
        output=str(tmp_path / "priority_out"),
        method_selection="all",
        audit_tau=[0.95],
        strict_units=True,
        tolerance=1e-12,
        allow_water_mass_volume_override=True,
    )

    run_summary = json.loads(Path(create_result.run_summary_json).read_text(encoding="utf-8"))
    priority_metadata = json.loads(Path(priority_result.flow_priority_metadata_json).read_text(encoding="utf-8"))

    assert run_summary["allow_water_mass_volume_override"] is True
    assert priority_metadata["allow_water_mass_volume_override"] is True


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
    assert run_summary["n_lcia_categories_used"] == 1
    with zipfile.ZipFile(result.output_zip, "r") as archive:
        assert "external_categories/category-1.json" not in archive.namelist()
        assert "external_methods/method-ext.json" not in archive.namelist()


def test_external_methods_units_are_merged_for_kg_to_g_conversion(tmp_path: Path):
    db_zip = make_database_with_support_objects_zip(tmp_path, name="external_g_support_db.zip")
    methods_zip = make_external_methods_zip_with_gram_units(tmp_path)

    create_result = create_command(
        database=str(db_zip),
        methods=str(methods_zip),
        output=str(tmp_path / "external_g_create_out"),
        tau=0.95,
        method_selection="all",
        uncharacterised_policy="keep",
        strict_units=True,
        tolerance=1e-12,
    )
    priority_result = priority_command(
        database=str(db_zip),
        methods=str(methods_zip),
        output=str(tmp_path / "external_g_priority_out"),
        method_selection="all",
        audit_tau=[0.95],
        strict_units=True,
        tolerance=1e-12,
    )

    run_summary = json.loads(Path(create_result.run_summary_json).read_text(encoding="utf-8"))
    priority_metadata = json.loads(Path(priority_result.flow_priority_metadata_json).read_text(encoding="utf-8"))

    assert run_summary["lcia_method_source"] == "external"
    assert run_summary["n_processes_error"] == 0
    assert priority_metadata["lcia_method_source"] == "external"
    assert priority_metadata["n_warning_records"] == 0


def test_external_methods_kg_to_g_conversion_works_with_unit_names_only(tmp_path: Path):
    db_zip = make_database_with_support_objects_zip(tmp_path, name="external_g_names_only_support_db.zip")
    methods_zip = make_external_methods_zip_with_gram_names_only(tmp_path)

    create_result = create_command(
        database=str(db_zip),
        methods=str(methods_zip),
        output=str(tmp_path / "external_g_names_only_create_out"),
        tau=0.95,
        method_selection="all",
        uncharacterised_policy="keep",
        strict_units=True,
        tolerance=1e-12,
    )
    priority_result = priority_command(
        database=str(db_zip),
        methods=str(methods_zip),
        output=str(tmp_path / "external_g_names_only_priority_out"),
        method_selection="all",
        audit_tau=[0.95],
        strict_units=True,
        tolerance=1e-12,
    )

    run_summary = json.loads(Path(create_result.run_summary_json).read_text(encoding="utf-8"))
    priority_metadata = json.loads(Path(priority_result.flow_priority_metadata_json).read_text(encoding="utf-8"))

    assert run_summary["lcia_method_source"] == "external"
    assert run_summary["n_processes_error"] == 0
    assert priority_metadata["lcia_method_source"] == "external"
    assert priority_metadata["n_warning_records"] == 0


def test_external_methods_kbq_to_bq_conversion_works_with_unit_names_only(tmp_path: Path):
    db_zip = make_database_with_activity_support_objects_zip(tmp_path, name="external_bq_names_only_support_db.zip")
    methods_zip = make_external_methods_zip_with_becquerel_names_only(tmp_path)

    create_result = create_command(
        database=str(db_zip),
        methods=str(methods_zip),
        output=str(tmp_path / "external_bq_names_only_create_out"),
        tau=0.95,
        method_selection="all",
        uncharacterised_policy="keep",
        strict_units=True,
        tolerance=1e-12,
    )
    priority_result = priority_command(
        database=str(db_zip),
        methods=str(methods_zip),
        output=str(tmp_path / "external_bq_names_only_priority_out"),
        method_selection="all",
        audit_tau=[0.95],
        strict_units=True,
        tolerance=1e-12,
    )

    run_summary = json.loads(Path(create_result.run_summary_json).read_text(encoding="utf-8"))
    priority_metadata = json.loads(Path(priority_result.flow_priority_metadata_json).read_text(encoding="utf-8"))

    assert run_summary["lcia_method_source"] == "external"
    assert run_summary["n_processes_error"] == 0
    assert priority_metadata["lcia_method_source"] == "external"
    assert priority_metadata["n_warning_records"] == 0


def test_create_preserves_database_support_objects_when_optional_methods_duplicate_them(tmp_path: Path):
    db_zip = make_database_with_support_objects_zip(tmp_path)
    methods_zip = make_methods_with_duplicate_support_objects_zip(tmp_path)
    output_dir = tmp_path / "duplicate_support_out"

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

    with zipfile.ZipFile(db_zip, "r") as database_archive:
        expected_flow_property = json.loads(database_archive.read("flow_properties/fp-mass.json"))
        expected_unit_group = json.loads(database_archive.read("unit_groups/ug-mass.json"))
    with zipfile.ZipFile(result.output_zip, "r") as output_archive:
        actual_flow_property = json.loads(output_archive.read("flow_properties/fp-mass.json"))
        actual_unit_group = json.loads(output_archive.read("unit_groups/ug-mass.json"))

    assert actual_flow_property == expected_flow_property
    assert actual_unit_group == expected_unit_group
    with zipfile.ZipFile(result.output_zip, "r") as output_archive:
        assert "lcia_methods/method-1.json" not in output_archive.namelist()
        assert "lcia_categories/category-1.json" not in output_archive.namelist()


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

    assert run_summary["lcia_method_source"] == "database"
    assert run_summary["internal_lcia_methods_ignored"] is False
    assert run_summary["n_lcia_categories_used"] == 2
    assert {row["method_name"] for row in inspect_json["lcia_categories"]} == {"Recovered Method"}


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
    assert run_summary["n_empty_lcia_categories"] == 1


def test_create_outputs_only_streaming_artifacts(tmp_path: Path):
    db_zip = make_toy_database_zip(tmp_path)
    output_dir = tmp_path / "streaming_only_out"

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

    names = {path.name for path in output_dir.iterdir() if not path.name.startswith(".")}
    assert names == {"reduced_database.zip", "run_summary.json", "reduction_debug.ndjson"}
    assert Path(result.output_zip).name == "reduced_database.zip"
    assert Path(result.reduction_debug_ndjson).name == "reduction_debug.ndjson"


def test_streamed_output_process_matches_direct_reduction(tmp_path: Path):
    db_zip = make_toy_database_zip(tmp_path)
    output_dir = tmp_path / "match_direct_out"

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

    archive_index = index_archive(str(db_zip), require_processes=True, require_flows=True)
    process_locator = next(iter(archive_index.processes.values()))
    with zipfile.ZipFile(db_zip, "r") as archive:
        process_data = parse_json_object(archive.read(process_locator.path), process_locator.path)
    expected = reduce_process(
        process_data=process_data,
        flow_lookup=archive_index.flows,
        categories=select_lcia_categories(list(archive_index.impact_categories.values()), "all"),
        tau=0.95,
        uncharacterised_policy="keep",
        strict_units=True,
        tol=1e-12,
        database_name=archive_index.source_name,
        unit_registry=archive_index.units,
    ).reduced_process
    actual = _read_zip_json(Path(result.output_zip), process_locator.path)

    assert actual == expected


def test_fail_fast_false_records_error_and_continues(tmp_path: Path):
    db_zip = make_fail_fast_database_zip(tmp_path)
    output_dir = tmp_path / "fail_fast_false_out"

    result = create_command(
        database=str(db_zip),
        methods=None,
        output=str(output_dir),
        tau=0.95,
        method_selection="all",
        uncharacterised_policy="keep",
        strict_units=True,
        tolerance=1e-12,
        fail_fast=False,
    )

    run_summary = json.loads(Path(result.run_summary_json).read_text(encoding="utf-8"))
    debug_lines = _read_debug_lines(Path(result.reduction_debug_ndjson))

    assert run_summary["n_processes_total"] == 2
    assert run_summary["n_processes_ok"] == 1
    assert run_summary["n_processes_error"] == 1
    assert [line["status"] for line in debug_lines] == ["error", "ok"]
    assert len(debug_lines) == 2

    bad_process = _read_zip_json(Path(result.output_zip), "processes/process-0-bad.json")
    good_process = _read_zip_json(Path(result.output_zip), "processes/process-1-good.json")
    assert bad_process["@id"] == "process-bad"
    assert [exchange["@id"] for exchange in good_process["exchanges"]] == ["product", "elem-1"]


def test_fail_fast_true_stops_on_first_error(tmp_path: Path):
    db_zip = make_fail_fast_database_zip(tmp_path)
    output_dir = tmp_path / "fail_fast_true_out"

    try:
        create_command(
            database=str(db_zip),
            methods=None,
            output=str(output_dir),
            tau=0.95,
            method_selection="all",
            uncharacterised_policy="keep",
            strict_units=True,
            tolerance=1e-12,
            fail_fast=True,
        )
    except MissingFlowError:
        pass
    else:
        raise AssertionError("Expected MissingFlowError")

    assert not (output_dir / "reduced_database.zip").exists()
    debug_lines = _read_debug_lines(output_dir / "reduction_debug.ndjson")
    run_summary = json.loads((output_dir / "run_summary.json").read_text(encoding="utf-8"))

    assert len(debug_lines) == 1
    assert debug_lines[0]["status"] == "error"
    assert run_summary["n_processes_error"] == 1


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


def test_match_priority_glad_cli_creates_outputs(tmp_path: Path, monkeypatch):
    priority_csv = _write_priority_match_csv(tmp_path / "priority.csv")
    asset_dir = _write_glad_asset_dir(tmp_path / "assets")
    output_dir = tmp_path / "out"

    priority_glad_module._load_target_dataset_cached.cache_clear()
    monkeypatch.setattr(priority_glad_module, "glad_asset_dir", lambda: asset_dir)

    exit_code = main(
        [
            "match-priority-glad",
            "--priority-csv",
            str(priority_csv),
            "--audit-tau",
            "0.95",
            "--output",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    assert (output_dir / "priority_glad_summary.json").exists()
    assert (output_dir / "priority_glad_summary.csv").exists()
    assert (output_dir / "ecoinventEFv3.7_matches.csv").exists()
