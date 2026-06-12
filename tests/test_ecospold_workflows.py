import csv
import json
import textwrap
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import pytest

from lci_reduce.cli import create_command, explore_ambiguities_command, priority_command
from lci_reduce.archive_reader import index_archive, load_archive
from lci_reduce.ecospold1_reader import iter_ecospold_processes
from lci_reduce.errors import DataFormatError


def _exchange_xml(
    *,
    number: int,
    name: str,
    unit: str,
    mean_value: float,
    category: str,
    subcategory: str,
    input_group: str = "",
    output_group: str = "",
) -> str:
    group_xml = ""
    if input_group:
        group_xml = f"<inputGroup>{input_group}</inputGroup>"
    elif output_group:
        group_xml = f"<outputGroup>{output_group}</outputGroup>"
    return (
        f'<exchange number="{number}" name="{name}" unit="{unit}" meanValue="{mean_value}" '
        f'category="{category}" subCategory="{subcategory}">{group_xml}</exchange>'
    )


def _process_dataset_xml() -> str:
    exchanges = "\n".join(
        [
            _exchange_xml(
                number=1,
                name="test product",
                unit="kg",
                mean_value=1.0,
                category="products",
                subcategory="test",
                output_group="0",
            ),
            _exchange_xml(
                number=2,
                name="Carbon dioxide, fossil",
                unit="kg",
                mean_value=1.0,
                category="air",
                subcategory="urban air",
                output_group="4",
            ),
            _exchange_xml(
                number=3,
                name="Methane, fossil",
                unit="kg",
                mean_value=100.0,
                category="air",
                subcategory="urban air",
                output_group="4",
            ),
            _exchange_xml(
                number=4,
                name="market electricity, medium voltage",
                unit="kWh",
                mean_value=2.0,
                category="technosphere",
                subcategory="grid",
                input_group="5",
            ),
            _exchange_xml(
                number=5,
                name="waste plastic, mixture",
                unit="kg",
                mean_value=0.25,
                category="waste",
                subcategory="treatment",
                output_group="3",
            ),
        ]
    )
    return textwrap.dedent(
        f"""\
        <?xml version="1.0" encoding="utf-8"?>
        <ecoSpold xmlns="http://www.EcoInvent.org/EcoSpold01" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
          <dataset number="1" timestamp="2026-06-04T12:00:00" generator="tests">
            <metaInformation>
              <processInformation>
                <referenceFunction
                  name="test process"
                  localName="test process"
                  datasetRelatesToProduct="true"
                  infrastructureProcess="false"
                  infrastructureIncluded="false"
                  amount="1"
                  unit="kg"
                  category="products"
                  subCategory="test"
                  localCategory="products"
                  localSubCategory="test"
                  includedProcesses=""
                  generalComment=""
                />
                <geography location="GLO" text="Global"/>
                <technology text="Test technology"/>
                <timePeriod text="2025" dataValidForEntirePeriod="true">
                  <startDate>2025-01-01</startDate>
                  <endDate>2025-12-31</endDate>
                </timePeriod>
                <dataSetInformation
                  type="1"
                  impactAssessmentResult="false"
                  timestamp="2026-06-04T12:00:00"
                  version="1.0"
                  internalVersion="1.0"
                  energyValues="0"
                  languageCode="en"
                  localLanguageCode="en"
                />
              </processInformation>
              <modellingAndValidation>
                <representativeness
                  productionVolume="1"
                  samplingProcedure=""
                  extrapolations=""
                  uncertaintyAdjustments=""
                />
              </modellingAndValidation>
              <administrativeInformation />
            </metaInformation>
            <flowData>
              {exchanges}
            </flowData>
          </dataset>
        </ecoSpold>
        """
    )


def _method_dataset_xml() -> str:
    factors = "\n".join(
        [
            _exchange_xml(
                number=1,
                name="Carbon dioxide, fossil",
                unit="kg",
                mean_value=1.0,
                category="air",
                subcategory="urban air",
            ),
            _exchange_xml(
                number=2,
                name="Methane, fossil",
                unit="kg",
                mean_value=1.0,
                category="air",
                subcategory="urban air",
            ),
        ]
    )
    return textwrap.dedent(
        f"""\
        <?xml version="1.0" encoding="utf-8"?>
        <ecoSpold xmlns="http://www.EcoInvent.org/EcoSpold01Impact" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
          <dataset number="1" timestamp="2026-06-04T12:00:00" generator="tests">
            <metaInformation>
              <processInformation>
                <referenceFunction
                  name="Global warming potential"
                  localName="Global warming potential"
                  datasetRelatesToProduct="false"
                  infrastructureProcess="false"
                  infrastructureIncluded="false"
                  amount="1"
                  unit="kg CO2 eq"
                  category="Test LCIA"
                  subCategory="Climate change"
                  localCategory="Test LCIA"
                  localSubCategory="Climate change"
                  includedProcesses=""
                  generalComment=""
                />
              </processInformation>
            </metaInformation>
            <flowData>
              {factors}
            </flowData>
          </dataset>
        </ecoSpold>
        """
    )


def _ambiguous_method_dataset_xml() -> str:
    factors = "\n".join(
        [
            _exchange_xml(
                number=1,
                name="Carbon dioxide, fossil",
                unit="kg",
                mean_value=1.0,
                category="air",
                subcategory="urban air",
            ),
            _exchange_xml(
                number=2,
                name="Methane, fossil",
                unit="kg",
                mean_value=1.0,
                category="air",
                subcategory="urban air",
            ),
            _exchange_xml(
                number=3,
                name="Methane, fossil",
                unit="kg",
                mean_value=2.0,
                category="air",
                subcategory="urban air",
            ),
        ]
    )
    return textwrap.dedent(
        f"""\
        <?xml version="1.0" encoding="utf-8"?>
        <ecoSpold xmlns="http://www.EcoInvent.org/EcoSpold01Impact" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
          <dataset number="1" timestamp="2026-06-04T12:00:00" generator="tests">
            <metaInformation>
              <processInformation>
                <referenceFunction
                  name="Ambiguous impact method"
                  localName="Ambiguous impact method"
                  datasetRelatesToProduct="false"
                  infrastructureProcess="false"
                  infrastructureIncluded="false"
                  amount="1"
                  unit="kg CO2 eq"
                  category="Test LCIA"
                  subCategory="Climate change"
                  localCategory="Test LCIA"
                  localSubCategory="Climate change"
                  includedProcesses=""
                  generalComment=""
                />
              </processInformation>
            </metaInformation>
            <flowData>
              {factors}
            </flowData>
          </dataset>
        </ecoSpold>
        """
    )


def _malformed_method_dataset_xml() -> str:
    return textwrap.dedent(
        """\
        <?xml version="1.0" encoding="utf-8"?>
        <ecoSpold xmlns="http://www.EcoInvent.org/EcoSpold01">
          <dataset number="1">
            <metaInformation>
              <processInformation>
                <referenceFunction name="Broken method" />
              </processInformation>
            </metaInformation>
            <flowData>
              <exchange number="1" name="Broken factor"
        """
    )


def _non_process_dataset_xml() -> str:
    return textwrap.dedent(
        """\
        <?xml version="1.0" encoding="utf-8"?>
        <ecoSpold xmlns="http://www.EcoInvent.org/EcoSpold01">
          <dataset number="1">
            <metaInformation>
              <processInformation>
                <referenceFunction
                  name="Documentation only"
                  datasetRelatesToProduct="false"
                  category="notes"
                  subCategory="text"
                />
              </processInformation>
            </metaInformation>
          </dataset>
        </ecoSpold>
        """
    )


def _write_zip(path: Path, files: dict[str, str]) -> Path:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for rel_path, text in files.items():
            archive.writestr(rel_path, text)
    return path


def _read_exchange_names(xml_bytes: bytes) -> list[str]:
    root = ET.fromstring(xml_bytes)
    ns = {"es": "http://www.EcoInvent.org/EcoSpold01"}
    return [element.get("name") or "" for element in root.findall(".//es:flowData/es:exchange", ns)]


def test_index_and_load_archive_detect_ecospold1(tmp_path: Path) -> None:
    database_zip = _write_zip(tmp_path / "ecospold_processes.zip", {"processes/test-process.xml": _process_dataset_xml()})

    index = index_archive(str(database_zip))
    archive = load_archive(str(database_zip))

    assert index.source_format == "ecospold1"
    assert archive.source_format == "ecospold1"
    assert len(index.processes) == 1
    assert any(flow.is_elementary for flow in index.flows.values())
    methane = next(flow for flow in index.flows.values() if flow.name == "Methane, fossil")
    assert methane.category_path == "air/urban air"


def test_index_archive_warn_skip_malformed_ecospold_methods_xml(tmp_path: Path) -> None:
    methods_zip = _write_zip(
        tmp_path / "ecospold_methods.zip",
        {
            "methods/gwp.xml": _method_dataset_xml(),
            "methods/broken-method.xml": _malformed_method_dataset_xml(),
        },
    )

    archive = index_archive(
        str(methods_zip),
        require_processes=False,
        require_flows=False,
        ecospold_xml_error_policy="warn_skip",
    )

    assert archive.source_format == "ecospold1"
    assert len(archive.impact_categories) == 1
    parse_warnings = archive.extra.get("parse_warnings", [])
    assert len(parse_warnings) == 1
    assert parse_warnings[0]["source_file"] == "methods/broken-method.xml"
    assert "Failed to parse EcoSpold1 XML file" in parse_warnings[0]["message"]


def test_ecospold_index_counts_only_semantic_process_datasets(tmp_path: Path) -> None:
    database_zip = _write_zip(
        tmp_path / "ecospold_mixed.zip",
        {
            "processes/test-process.xml": _process_dataset_xml(),
            "processes/non-process.xml": _non_process_dataset_xml(),
            "processes/broken.xml": _malformed_method_dataset_xml(),
        },
    )

    index = index_archive(
        str(database_zip),
        ecospold_xml_error_policy="warn_skip",
    )
    archive = load_archive(
        str(database_zip),
        ecospold_xml_error_policy="warn_skip",
    )
    iterated = list(iter_ecospold_processes(index))

    assert index.source_format == "ecospold1"
    assert len(index.processes) == 1
    assert len(archive.processes) == 1
    assert len(iterated) == 1
    assert next(iter(index.processes.values())).name == "test process"
    parse_warnings = index.extra.get("parse_warnings", [])
    assert len(parse_warnings) == 2
    assert {warning["source_file"] for warning in parse_warnings} == {"processes/non-process.xml", "processes/broken.xml"}
    assert any("Skipped non-process EcoSpold1 dataset" in warning["message"] for warning in parse_warnings)


def test_load_archive_accepts_single_ecospold_method_xml_file(tmp_path: Path) -> None:
    method_xml = tmp_path / "single-method.xml"
    method_xml.write_text(_method_dataset_xml(), encoding="utf-8")

    archive = load_archive(str(method_xml), require_processes=False, require_flows=False)

    assert archive.source_format == "ecospold1"
    assert len(archive.impact_categories) == 1
    assert len(archive.impact_methods) == 1
    assert len(archive.processes) == 0
    category = next(iter(archive.impact_categories.values()))
    assert category.method_name == "Test LCIA"
    assert category.name == "Global warming potential"


def test_database_archive_still_fails_on_malformed_ecospold_process_xml(tmp_path: Path) -> None:
    database_zip = _write_zip(
        tmp_path / "ecospold_processes.zip",
        {"processes/broken-process.xml": _malformed_method_dataset_xml()},
    )

    with pytest.raises(DataFormatError, match="Failed to parse EcoSpold1 XML file"):
        index_archive(str(database_zip))


def test_create_command_fails_when_no_usable_lcia_categories_are_available(tmp_path: Path) -> None:
    database_zip = _write_zip(tmp_path / "ecospold_processes.zip", {"processes/test-process.xml": _process_dataset_xml()})
    methods_zip = _write_zip(
        tmp_path / "ecospold_methods.zip",
        {"methods/broken-method.xml": _malformed_method_dataset_xml()},
    )
    output_dir = tmp_path / "out"

    with pytest.raises(DataFormatError, match="No usable LCIA categories were available for this run"):
        create_command(
            database=str(database_zip),
            methods=str(methods_zip),
            output=str(output_dir),
            tau=0.95,
            method_selection="all",
            uncharacterised_policy="drop",
            strict_units=True,
            tolerance=1e-12,
        )


def test_priority_command_fails_when_no_usable_lcia_categories_are_available(tmp_path: Path) -> None:
    database_zip = _write_zip(tmp_path / "ecospold_processes.zip", {"processes/test-process.xml": _process_dataset_xml()})
    methods_zip = _write_zip(
        tmp_path / "ecospold_methods.zip",
        {"methods/broken-method.xml": _malformed_method_dataset_xml()},
    )
    output_dir = tmp_path / "priority_out"

    with pytest.raises(DataFormatError, match="No usable LCIA categories were available for this priority run"):
        priority_command(
            database=str(database_zip),
            methods=str(methods_zip),
            output=str(output_dir),
            method_selection="all",
            audit_tau=[0.95, 0.99],
            strict_units=True,
            tolerance=1e-12,
        )


def test_create_command_reduces_ecospold_archive_and_preserves_non_elementary_exchanges(tmp_path: Path) -> None:
    database_zip = _write_zip(tmp_path / "ecospold_processes.zip", {"processes/test-process.xml": _process_dataset_xml()})
    methods_zip = _write_zip(
        tmp_path / "ecospold_methods.zip",
        {
            "methods/gwp.xml": _method_dataset_xml(),
            "methods/broken-method.xml": _malformed_method_dataset_xml(),
        },
    )
    output_dir = tmp_path / "out"

    result = create_command(
        database=str(database_zip),
        methods=str(methods_zip),
        output=str(output_dir),
        tau=0.95,
        method_selection="all",
        uncharacterised_policy="keep",
        strict_units=True,
        tolerance=1e-12,
    )

    summary = json.loads(Path(result.run_summary_json).read_text(encoding="utf-8"))
    assert summary["source_format"] == "ecospold1"
    assert summary["n_elementary_removed"] == 1
    assert summary["n_input_parse_warnings"] == 1
    assert summary["input_parse_warnings"][0]["source_file"] == "methods/broken-method.xml"

    with zipfile.ZipFile(result.output_zip, "r") as archive:
        xml_bytes = archive.read("processes/test-process.xml")
    exchange_names = _read_exchange_names(xml_bytes)
    assert "Carbon dioxide, fossil" not in exchange_names
    assert "Methane, fossil" in exchange_names
    assert "test product" in exchange_names
    assert "market electricity, medium voltage" in exchange_names
    assert "waste plastic, mixture" in exchange_names


def test_priority_command_generates_ecospold_priority_sidecars(tmp_path: Path) -> None:
    database_zip = _write_zip(tmp_path / "ecospold_processes.zip", {"processes/test-process.xml": _process_dataset_xml()})
    methods_zip = _write_zip(
        tmp_path / "ecospold_methods.zip",
        {
            "methods/gwp.xml": _method_dataset_xml(),
            "methods/broken-method.xml": _malformed_method_dataset_xml(),
        },
    )
    output_dir = tmp_path / "priority_out"

    result = priority_command(
        database=str(database_zip),
        methods=str(methods_zip),
        output=str(output_dir),
        method_selection="all",
        audit_tau=[0.95, 0.99],
        strict_units=True,
        tolerance=1e-12,
    )

    metadata = json.loads(Path(result.flow_priority_metadata_json).read_text(encoding="utf-8"))
    assert metadata["source_format"] == "ecospold1"
    assert metadata["n_processes_total"] == 1
    assert any(
        warning.get("source_file") == "methods/broken-method.xml"
        for warning in metadata.get("warnings", [])
    )

    with Path(result.flow_priority_csv).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    names = {row["flow_name"] for row in rows}
    assert names == {"Carbon dioxide, fossil", "Methane, fossil"}
    for row in rows:
        assert row["compartment"] == "air"
        assert row["subcompartment"] == "urban air"
        assert row["occurrence_count"] == "1"
        assert row["characterised_occurrence_count"] == "1"


def test_explore_ambiguities_command_scans_ecospold1_archives(tmp_path: Path) -> None:
    database_zip = _write_zip(tmp_path / "ecospold_processes.zip", {"processes/test-process.xml": _process_dataset_xml()})
    methods_zip = _write_zip(
        tmp_path / "ecospold_methods.zip",
        {
            "methods/ambiguous.xml": _ambiguous_method_dataset_xml(),
        },
    )
    output_dir = tmp_path / "ambiguity_out"

    result = explore_ambiguities_command(
        database=str(database_zip),
        methods=str(methods_zip),
        output=str(output_dir),
        method_selection="all",
        strict_units=True,
        tolerance=1e-12,
    )

    metadata = json.loads(Path(result.cf_ambiguity_metadata_json).read_text(encoding="utf-8"))
    assert metadata["source_format"] == "ecospold1"
    assert metadata["csv_row_mode"] == "group_summary"
    assert metadata["scan_mode"] == "record_only_first_n_processes"
    assert metadata["process_scan_limit"] == 1
    assert metadata["n_processes_total"] == 1
    assert metadata["n_cf_ambiguity_records"] > 0
    assert metadata["n_non_location_ambiguity_groups"] >= 1
    assert metadata["n_method_mixed_groups"] == 0
    assert Path(result.cf_ambiguities_csv).exists()
