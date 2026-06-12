from __future__ import annotations

import csv
import json
import textwrap
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import pytest

from lca_flowkit import analyse_flows, create_priority_file, reduce_database
from lca_flowkit.errors import DataFormatError
from lca_flowkit.inputs import load_bundle
from lci_reduce.cli import create_command, priority_command


def _write_json_zip(path: Path, files: dict[str, dict]) -> Path:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for rel_path, data in files.items():
            archive.writestr(rel_path, json.dumps(data))
    return path


def _read_json_process(zip_path: Path, rel_path: str) -> dict:
    with zipfile.ZipFile(zip_path, "r") as archive:
        return json.loads(archive.read(rel_path).decode("utf-8"))


def _simple_jsonld_db(base: Path) -> Path:
    return _write_json_zip(
        base / "db.zip",
        {
            "processes/p1.json": {
                "@id": "p1",
                "@type": "Process",
                "name": "P1",
                "location": {"name": "GB"},
                "exchanges": [
                    {"@id": "prod", "amount": 1.0, "flow": {"@id": "fp", "name": "Product"}, "unit": {"name": "kg"}, "quantitativeReference": True},
                    {"@id": "co2", "amount": 10.0, "flow": {"@id": "f-co2", "name": "CO2"}, "unit": {"name": "kg"}},
                    {"@id": "ch4", "amount": 0.1, "flow": {"@id": "f-ch4", "name": "CH4"}, "unit": {"name": "kg"}},
                    {"@id": "unk", "amount": 5.0, "flow": {"@id": "f-unk", "name": "Uncharacterised"}, "unit": {"name": "kg"}},
                ],
            },
            "flows/fp.json": {"@id": "fp", "@type": "Flow", "name": "Product", "flowType": "PRODUCT_FLOW", "categoryPath": "products"},
            "flows/f-co2.json": {"@id": "f-co2", "@type": "Flow", "name": "CO2", "flowType": "ELEMENTARY_FLOW", "categoryPath": "air/urban air"},
            "flows/f-ch4.json": {"@id": "f-ch4", "@type": "Flow", "name": "CH4", "flowType": "ELEMENTARY_FLOW", "categoryPath": "air/urban air"},
            "flows/f-unk.json": {"@id": "f-unk", "@type": "Flow", "name": "Uncharacterised", "flowType": "ELEMENTARY_FLOW", "categoryPath": "air/urban air"},
            "lcia_methods/m1.json": {"@id": "m1", "@type": "ImpactMethod", "name": "IPCC"},
            "lcia_categories/c1.json": {
                "@id": "c1",
                "@type": "ImpactCategory",
                "name": "Climate change",
                "impactMethod": {"@id": "m1", "name": "IPCC"},
                "referenceUnitName": "kg CO2e",
                "impactFactors": [
                    {"flow": {"@id": "f-co2", "name": "CO2"}, "value": 1.0, "unitName": "kg"},
                    {"flow": {"@id": "f-ch4", "name": "CH4"}, "value": 1.0, "unitName": "kg"},
                ],
            },
        },
    )


def _jsonld_db_with_sidecar_paths(base: Path, *, root_prefix: str = "") -> Path:
    return _write_json_zip(
        base / "db_with_sidecars.zip",
        {
            (f"{root_prefix.rstrip('/')}/processes/p1.json" if root_prefix else "processes/p1.json"): {
                "@id": "p1",
                "@type": "Process",
                "name": "P1",
                "location": {"name": "GB"},
                "exchanges": [
                    {"@id": "prod", "amount": 1.0, "flow": {"@id": "fp", "name": "Product"}, "unit": {"name": "kg"}, "quantitativeReference": True},
                    {"@id": "co2", "amount": 10.0, "flow": {"@id": "f-co2", "name": "CO2"}, "unit": {"name": "kg"}},
                ],
            },
            (f"{root_prefix.rstrip('/')}/bin/processes/p1/calculation-preferences.json" if root_prefix else "bin/processes/p1/calculation-preferences.json"): {
                "@id": "process-sidecar",
                "@type": "Process",
                "name": "Calculation preferences",
                "solver": "deterministic",
            },
            (f"{root_prefix.rstrip('/')}/flows/fp.json" if root_prefix else "flows/fp.json"): {"@id": "fp", "@type": "Flow", "name": "Product", "flowType": "PRODUCT_FLOW", "categoryPath": "products"},
            (f"{root_prefix.rstrip('/')}/flows/f-co2.json" if root_prefix else "flows/f-co2.json"): {"@id": "f-co2", "@type": "Flow", "name": "CO2", "flowType": "ELEMENTARY_FLOW", "categoryPath": "air/urban air"},
            (f"{root_prefix.rstrip('/')}/flow_properties/fp-mass.json" if root_prefix else "flow_properties/fp-mass.json"): {"@id": "fp-mass", "@type": "FlowProperty", "name": "Mass"},
            (f"{root_prefix.rstrip('/')}/lcia_methods/m1.json" if root_prefix else "lcia_methods/m1.json"): {"@id": "m1", "@type": "ImpactMethod", "name": "IPCC"},
            (f"{root_prefix.rstrip('/')}/lcia_categories/c1.json" if root_prefix else "lcia_categories/c1.json"): {
                "@id": "c1",
                "@type": "ImpactCategory",
                "name": "Climate change",
                "impactMethod": {"@id": "m1", "name": "IPCC"},
                "referenceUnitName": "kg CO2e",
                "impactFactors": [
                    {"flow": {"@id": "f-co2", "name": "CO2"}, "value": 1.0, "unitName": "kg"},
                ],
            },
        },
    )


def _read_priority_rows(csv_path: Path) -> dict[str, dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return {row["flow_id"]: row for row in csv.DictReader(handle)}


def test_reduce_database_keeps_uncharacterised_by_default(tmp_path: Path) -> None:
    db = _simple_jsonld_db(tmp_path)
    result = reduce_database(str(db), output_dir=str(tmp_path / "reduced"), tau=0.95)
    process = _read_json_process(Path(result.output_zip), "processes/p1.json")
    exchange_ids = [exchange["@id"] for exchange in process["exchanges"]]
    assert "ch4" not in exchange_ids
    assert "co2" in exchange_ids
    assert "unk" in exchange_ids


def test_priority_and_reducer_match_legacy_on_simple_jsonld(tmp_path: Path) -> None:
    db = _simple_jsonld_db(tmp_path)
    clean_reduce = reduce_database(str(db), output_dir=str(tmp_path / "clean_reduce"), tau=0.95)
    legacy_reduce = create_command(
        database=str(db),
        methods=None,
        output=str(tmp_path / "legacy_reduce"),
        tau=0.95,
        method_selection="all",
        uncharacterised_policy="keep",
        strict_units=True,
        tolerance=1e-12,
    )
    clean_process = _read_json_process(Path(clean_reduce.output_zip), "processes/p1.json")
    legacy_process = _read_json_process(Path(legacy_reduce.output_zip), "processes/p1.json")
    assert [exchange["@id"] for exchange in clean_process["exchanges"]] == [exchange["@id"] for exchange in legacy_process["exchanges"]]

    clean_priority = create_priority_file(str(db), output_dir=str(tmp_path / "clean_priority"), audit_tau=[0.95])
    legacy_priority = priority_command(
        database=str(db),
        methods=None,
        output=str(tmp_path / "legacy_priority"),
        method_selection="all",
        audit_tau=[0.95],
        strict_units=True,
        tolerance=1e-12,
    )
    clean_rows = _read_priority_rows(Path(clean_priority.priority_csv))
    legacy_rows = _read_priority_rows(Path(legacy_priority.flow_priority_csv))
    assert set(clean_rows) == set(legacy_rows)
    for flow_id in {"f-co2", "f-ch4"}:
        assert float(clean_rows[flow_id]["eta_0_95"]) == pytest.approx(float(legacy_rows[flow_id]["eta_0_95"]))
        assert float(clean_rows[flow_id]["loss_max_0_95"]) == pytest.approx(float(legacy_rows[flow_id]["loss_max_0_95"]))


def test_jsonld_sidecar_paths_are_not_treated_as_processes_or_flows(tmp_path: Path) -> None:
    db = _jsonld_db_with_sidecar_paths(tmp_path)
    bundle = load_bundle(str(db))
    reduction = reduce_database(str(db), output_dir=str(tmp_path / "sidecar_reduce"), tau=0.95)
    priority = create_priority_file(str(db), output_dir=str(tmp_path / "sidecar_priority"), audit_tau=[0.95])

    assert len(bundle.processes) == 1
    assert set(bundle.flows) == {"fp", "f-co2"}
    sidecar_entry = next(entry for entry in bundle.entries if entry.path == "bin/processes/p1/calculation-preferences.json")
    flow_property_entry = next(entry for entry in bundle.entries if entry.path == "flow_properties/fp-mass.json")
    assert sidecar_entry.object_type == "other"
    assert flow_property_entry.object_type == "flow_property"
    assert reduction.counts["n_processes_total"] == 1
    assert priority.counts["n_processes_total"] == 1

    with zipfile.ZipFile(db, "r") as source_archive, zipfile.ZipFile(reduction.output_zip, "r") as output_archive:
        sidecar_path = "bin/processes/p1/calculation-preferences.json"
        assert json.loads(output_archive.read(sidecar_path).decode("utf-8")) == json.loads(source_archive.read(sidecar_path).decode("utf-8"))
        assert "exchanges" not in json.loads(output_archive.read(sidecar_path).decode("utf-8"))


def test_root_prefixed_jsonld_sidecar_paths_are_not_treated_as_processes_or_flows(tmp_path: Path) -> None:
    root_prefix = "BAFU_export"
    db = _jsonld_db_with_sidecar_paths(tmp_path, root_prefix=root_prefix)
    bundle = load_bundle(str(db))
    reduction = reduce_database(str(db), output_dir=str(tmp_path / "rooted_sidecar_reduce"), tau=0.95)

    assert len(bundle.processes) == 1
    assert set(bundle.flows) == {"fp", "f-co2"}
    sidecar_path = f"{root_prefix}/bin/processes/p1/calculation-preferences.json"
    sidecar_entry = next(entry for entry in bundle.entries if entry.path == sidecar_path)
    assert sidecar_entry.object_type == "other"
    assert reduction.counts["n_processes_total"] == 1

    with zipfile.ZipFile(db, "r") as source_archive, zipfile.ZipFile(reduction.output_zip, "r") as output_archive:
        assert json.loads(output_archive.read(sidecar_path).decode("utf-8")) == json.loads(source_archive.read(sidecar_path).decode("utf-8"))


def _process_xml() -> str:
    return textwrap.dedent(
        """\
        <?xml version="1.0" encoding="utf-8"?>
        <ecoSpold xmlns="http://www.EcoInvent.org/EcoSpold01">
          <dataset number="1">
            <metaInformation>
              <processInformation>
                <referenceFunction name="test process" unit="kg" category="products" subCategory="test" />
                <geography location="GLO" text="Global"/>
              </processInformation>
            </metaInformation>
            <flowData>
              <exchange number="1" name="test product" unit="kg" meanValue="1.0" category="products" subCategory="test"><outputGroup>0</outputGroup></exchange>
              <exchange number="2" name="Carbon dioxide, fossil" unit="kg" meanValue="1.0" category="air" subCategory="urban air"><outputGroup>4</outputGroup></exchange>
              <exchange number="3" name="Methane, fossil" unit="kg" meanValue="100.0" category="air" subCategory="urban air"><outputGroup>4</outputGroup></exchange>
            </flowData>
          </dataset>
        </ecoSpold>
        """
    )


def _method_xml() -> str:
    return textwrap.dedent(
        """\
        <?xml version="1.0" encoding="utf-8"?>
        <ecoSpold xmlns="http://www.EcoInvent.org/EcoSpold01Impact">
          <dataset number="1">
            <metaInformation>
              <processInformation>
                <referenceFunction name="Global warming potential" unit="kg CO2e" category="Test LCIA" subCategory="Climate change" />
              </processInformation>
            </metaInformation>
            <flowData>
              <exchange number="1" name="Carbon dioxide, fossil" unit="kg" meanValue="1.0" category="air" subCategory="urban air" />
              <exchange number="2" name="Methane, fossil" unit="kg" meanValue="1.0" category="air" subCategory="urban air" />
            </flowData>
          </dataset>
        </ecoSpold>
        """
    )


def _non_process_xml() -> str:
    return textwrap.dedent(
        """\
        <?xml version="1.0" encoding="utf-8"?>
        <ecoSpold xmlns="http://www.EcoInvent.org/EcoSpold01">
          <dataset number="1">
            <metaInformation>
              <processInformation>
                <referenceFunction name="Documentation only" datasetRelatesToProduct="false" category="notes" subCategory="text" />
              </processInformation>
            </metaInformation>
          </dataset>
        </ecoSpold>
        """
    )


def test_malformed_ecospold_xml_fails_cleanly(tmp_path: Path) -> None:
    broken = tmp_path / "broken.zip"
    with zipfile.ZipFile(broken, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("processes/broken.xml", "<ecoSpold><dataset>")
    with pytest.raises(DataFormatError, match="Failed to parse EcoSpold1 XML file"):
        load_bundle(str(broken))


def test_reduce_and_prioritise_ecospold1_archive(tmp_path: Path) -> None:
    db = tmp_path / "ecospold_db.zip"
    methods = tmp_path / "ecospold_methods.zip"
    with zipfile.ZipFile(db, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("processes/test-process.xml", _process_xml())
    with zipfile.ZipFile(methods, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("methods/gwp.xml", _method_xml())

    reduction = reduce_database(str(db), methods_path=str(methods), output_dir=str(tmp_path / "eco_reduce"), tau=0.95)
    priority = create_priority_file(str(db), methods_path=str(methods), output_dir=str(tmp_path / "eco_priority"), audit_tau=[0.95])
    with zipfile.ZipFile(reduction.output_zip, "r") as archive:
        root = ET.fromstring(archive.read("processes/test-process.xml"))
    ns = {"es": "http://www.EcoInvent.org/EcoSpold01"}
    names = [element.get("name") for element in root.findall(".//es:flowData/es:exchange", ns)]
    assert "Carbon dioxide, fossil" not in names
    assert "Methane, fossil" in names
    report = analyse_flows(priority.priority_csv, priority.metadata_json, flow_names=["Methane, fossil"], tau=0.95)
    assert report.matched_flow_ids


def test_ecospold_bundle_skips_non_process_xml_when_counting_processes(tmp_path: Path) -> None:
    db = tmp_path / "ecospold_mixed.zip"
    methods = tmp_path / "ecospold_methods.zip"
    with zipfile.ZipFile(db, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("processes/test-process.xml", _process_xml())
        archive.writestr("processes/non-process.xml", _non_process_xml())
    with zipfile.ZipFile(methods, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("methods/gwp.xml", _method_xml())

    bundle = load_bundle(str(db))
    reduction = reduce_database(str(db), methods_path=str(methods), output_dir=str(tmp_path / "mixed_reduce"), tau=0.95)

    assert bundle.source_format == "ecospold1"
    assert len(bundle.processes) == 1
    assert bundle.extra.get("parse_warnings")
    assert reduction.counts["n_processes_total"] == 1
