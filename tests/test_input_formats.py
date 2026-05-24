import json
import shutil
import zipfile
from pathlib import Path

import pytest

from lci_reduce.errors import DataFormatError
from lci_reduce.jsonld_reader import load_archive


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

    monkeypatch.setattr("lci_reduce.jsonld_reader.get_cached_native_archive_jsonld", fake_convert)

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

    monkeypatch.setattr("lci_reduce.jsonld_reader.get_cached_native_archive_jsonld", fake_convert)

    with pytest.raises(DataFormatError, match="conversion failed"):
        load_archive(str(database_zip))
