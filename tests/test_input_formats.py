import json
import shutil
import zipfile
from pathlib import Path

import pytest

from lci_reduce.errors import DataFormatError
from lci_reduce.archive_reader import index_archive, load_archive


def _write_json_archive_folder(base: Path, files: dict[str, dict]) -> Path:
    for rel_path, data in files.items():
        target = base / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(data), encoding="utf-8")
    return base


def _write_json_archive_zip(path: Path, files: dict[str, dict]) -> Path:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for rel_path, data in files.items():
            archive.writestr(rel_path, json.dumps(data))
    return path


def test_non_object_json_fails_clearly(tmp_path: Path):
    database_zip = tmp_path / "bad.zip"
    with zipfile.ZipFile(database_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("processes/process-1.json", "[]")
    with pytest.raises(DataFormatError, match="top-level JSON object"):
        load_archive(str(database_zip))


def test_utf8_bom_json_is_accepted(tmp_path: Path):
    database_zip = tmp_path / "bom.zip"
    process_bytes = ("\ufeff" + json.dumps({"@id": "p1", "@type": "Process", "name": "P", "exchanges": []})).encode(
        "utf-8"
    )
    flow_bytes = ("\ufeff" + json.dumps({"@id": "f1", "@type": "Flow", "name": "F", "flowType": "PRODUCT_FLOW"})).encode(
        "utf-8"
    )
    with zipfile.ZipFile(database_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("processes/process-1.json", process_bytes)
        archive.writestr("flows/flow-1.json", flow_bytes)

    archive = load_archive(str(database_zip))
    assert len(archive.processes) == 1
    assert len(archive.flows) == 1


def test_sip_input_fails_clearly(tmp_path: Path):
    source = tmp_path / "input.sip"
    source.write_text("not supported", encoding="utf-8")
    with pytest.raises(DataFormatError, match=r"SimaPro \.sip packages are not supported"):
        load_archive(str(source))


def test_native_openlca_backup_zip_is_converted_when_converter_is_available(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    database_zip = tmp_path / "native_backup.zip"
    with zipfile.ZipFile(database_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("service.properties", "derby.serviceProtocol=org.apache.derby.database.Database")
        archive.writestr("README_DO_NOT_TOUCH_FILES.txt", "DERBY DATABASE")
        archive.writestr("log/README_DO_NOT_TOUCH_FILES.txt", "DERBY LOG")
        archive.writestr("seg0/c10.dat", b"not jsonld")

    converted_dir = tmp_path / "converted"
    converted_dir.mkdir()
    converted_zip = converted_dir / "converted.zip"
    with zipfile.ZipFile(converted_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "processes/process-1.json",
            json.dumps({"@id": "p1", "@type": "Process", "name": "P", "exchanges": []}),
        )
        archive.writestr(
            "flows/flow-1.json",
            json.dumps({"@id": "f1", "@type": "Flow", "name": "F", "flowType": "PRODUCT_FLOW"}),
        )

    def fake_convert(source: str, *, scope: str = "full") -> str:
        assert source == str(database_zip)
        assert scope == "full"
        temp_dir = tmp_path / "temp-export"
        temp_dir.mkdir()
        temp_zip = temp_dir / "converted.zip"
        shutil.copy2(converted_zip, temp_zip)
        return str(temp_zip)

    monkeypatch.setattr("lci_reduce.archive_reader.get_cached_native_archive_jsonld", fake_convert)

    archive = load_archive(str(database_zip))
    assert len(archive.processes) == 1
    assert len(archive.flows) == 1


def test_native_openlca_backup_zip_fails_clearly_when_conversion_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    database_zip = tmp_path / "native_backup.zip"
    with zipfile.ZipFile(database_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("service.properties", "derby.serviceProtocol=org.apache.derby.database.Database")
        archive.writestr("README_DO_NOT_TOUCH_FILES.txt", "DERBY DATABASE")
        archive.writestr("log/README_DO_NOT_TOUCH_FILES.txt", "DERBY LOG")
        archive.writestr("seg0/c10.dat", b"not jsonld")

    def fake_convert(_: str, *, scope: str = "full") -> str:
        assert scope == "full"
        raise DataFormatError("conversion failed")

    monkeypatch.setattr("lci_reduce.archive_reader.get_cached_native_archive_jsonld", fake_convert)

    with pytest.raises(DataFormatError, match="conversion failed"):
        load_archive(str(database_zip))


def test_load_archive_keeps_explicit_flow_category_path_for_folder_sources(tmp_path: Path) -> None:
    database_dir = _write_json_archive_folder(
        tmp_path / "folder_db",
        {
            "processes/process-1.json": {
                "@id": "process-1",
                "@type": "Process",
                "name": "P",
                "exchanges": [
                    {
                        "@id": "elem-1",
                        "amount": 1.0,
                        "flow": {"@id": "flow-co2", "name": "CO2"},
                    }
                ],
            },
            "flows/flow-co2.json": {
                "@id": "flow-co2",
                "@type": "Flow",
                "name": "CO2",
                "flowType": "ELEMENTARY_FLOW",
                "categoryPath": "air/urban air",
            },
        },
    )

    archive = load_archive(str(database_dir))

    assert archive.flows["flow-co2"].category_path == "air/urban air"
    assert archive.category_path_diagnostics.n_category_objects_indexed == 0
    assert archive.category_path_diagnostics.n_elementary_flows == 1
    assert archive.category_path_diagnostics.n_elementary_flows_with_category_path == 1
    assert archive.category_path_diagnostics.pct_elementary_flows_with_category_path == pytest.approx(100.0)


def test_index_archive_resolves_flow_category_parent_chain_from_zip(tmp_path: Path) -> None:
    database_zip = _write_json_archive_zip(
        tmp_path / "category_chain.zip",
        {
            "processes/process-1.json": {
                "@id": "process-1",
                "@type": "Process",
                "name": "P",
                "exchanges": [
                    {
                        "@id": "elem-1",
                        "amount": 1.0,
                        "flow": {"@id": "flow-co2", "name": "CO2"},
                    }
                ],
            },
            "flows/flow-co2.json": {
                "@id": "flow-co2",
                "@type": "Flow",
                "name": "CO2",
                "flowType": "ELEMENTARY_FLOW",
                "category": {"@id": "cat-urban-air", "name": "urban air"},
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
                "parentCategory": {"@id": "cat-air", "name": "air"},
            },
        },
    )

    archive = index_archive(str(database_zip))

    assert archive.flows["flow-co2"].category_path == "air/urban air"
    assert archive.category_path_diagnostics.n_category_objects_indexed == 2
    assert archive.category_path_diagnostics.n_category_paths_resolved == 2
    assert archive.category_path_diagnostics.n_elementary_flows_with_category_path == 1


def test_index_archive_matches_category_aliases_between_ref_id_and_at_id(tmp_path: Path) -> None:
    database_zip = _write_json_archive_zip(
        tmp_path / "category_aliases.zip",
        {
            "processes/process-1.json": {
                "@id": "process-1",
                "@type": "Process",
                "name": "P",
                "exchanges": [
                    {
                        "@id": "elem-1",
                        "amount": 1.0,
                        "flow": {"@id": "flow-co2", "name": "CO2"},
                    }
                ],
            },
            "flows/flow-co2.json": {
                "@id": "flow-co2",
                "@type": "Flow",
                "name": "CO2",
                "flowType": "ELEMENTARY_FLOW",
                "category": {"refId": "cat-urban-air-ref", "name": "urban air"},
            },
            "flow_categories/cat-air.json": {
                "@id": "urn:category:air",
                "refId": "cat-air-ref",
                "@type": "Category",
                "name": "air",
            },
            "flow_categories/cat-urban-air.json": {
                "@id": "urn:category:urban-air",
                "refId": "cat-urban-air-ref",
                "@type": "Category",
                "name": "urban air",
                "parentCategory": {"refId": "cat-air-ref", "name": "air"},
            },
        },
    )

    archive = index_archive(str(database_zip))

    assert archive.flows["flow-co2"].category_path == "air/urban air"
    assert archive.category_path_diagnostics.n_category_objects_indexed == 2
    assert archive.category_path_diagnostics.n_category_identifier_aliases_indexed == 4
    assert archive.category_path_diagnostics.n_flows_with_category_reference == 1
    assert archive.category_path_diagnostics.n_flows_with_resolved_category_reference == 1


def test_jsonld_indexing_ignores_sidecar_process_paths_and_flow_properties(tmp_path: Path) -> None:
    database_zip = _write_json_archive_zip(
        tmp_path / "sidecars.zip",
        {
            "processes/process-1.json": {
                "@id": "process-1",
                "@type": "Process",
                "name": "Real process",
                "exchanges": [],
            },
            "bin/processes/process-1/calculation-preferences.json": {
                "@id": "process-sidecar",
                "@type": "Process",
                "name": "Calculation preferences",
                "solver": "deterministic",
            },
            "flows/flow-1.json": {
                "@id": "flow-1",
                "@type": "Flow",
                "name": "Product",
                "flowType": "PRODUCT_FLOW",
            },
            "flow_properties/fp-mass.json": {
                "@id": "fp-mass",
                "@type": "FlowProperty",
                "name": "Mass",
            },
        },
    )

    index = index_archive(str(database_zip))
    archive = load_archive(str(database_zip))

    assert set(index.processes) == {"process-1"}
    assert set(index.flows) == {"flow-1"}
    assert set(archive.processes) == {"process-1"}
    assert set(archive.flows) == {"flow-1"}
    sidecar_entry = next(entry for entry in archive.other_entries if entry.path == "bin/processes/process-1/calculation-preferences.json")
    flow_property_entry = next(entry for entry in archive.other_entries if entry.path == "flow_properties/fp-mass.json")
    assert sidecar_entry.object_type == "other"
    assert flow_property_entry.object_type == "flow_property"


def test_jsonld_indexing_ignores_root_prefixed_sidecar_process_paths(tmp_path: Path) -> None:
    root_prefix = "BAFU_export"
    database_zip = _write_json_archive_zip(
        tmp_path / "rooted_sidecars.zip",
        {
            f"{root_prefix}/processes/process-1.json": {
                "@id": "process-1",
                "@type": "Process",
                "name": "Real process",
                "exchanges": [],
            },
            f"{root_prefix}/bin/processes/process-1/calculation-preferences.json": {
                "@id": "process-sidecar",
                "@type": "Process",
                "name": "Calculation preferences",
                "solver": "deterministic",
            },
            f"{root_prefix}/flows/flow-1.json": {
                "@id": "flow-1",
                "@type": "Flow",
                "name": "Product",
                "flowType": "PRODUCT_FLOW",
            },
        },
    )

    index = index_archive(str(database_zip))
    archive = load_archive(str(database_zip))

    assert set(index.processes) == {"process-1"}
    assert set(index.flows) == {"flow-1"}
    assert set(archive.processes) == {"process-1"}
    sidecar_entry = next(
        entry
        for entry in archive.other_entries
        if entry.path == f"{root_prefix}/bin/processes/process-1/calculation-preferences.json"
    )
    assert sidecar_entry.object_type == "other"
