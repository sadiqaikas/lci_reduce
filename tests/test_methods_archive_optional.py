import json
import zipfile
from pathlib import Path

from lci_reduce.jsonld_reader import load_archive


def test_methods_archive_can_be_methods_only(tmp_path: Path):
    methods_zip = tmp_path / "methods_only.zip"
    with zipfile.ZipFile(methods_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
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
                            "value": 1.0,
                            "unitName": "kg",
                        }
                    ],
                }
            ),
        )
    archive = load_archive(str(methods_zip), require_processes=False, require_flows=False)
    assert len(archive.impact_categories) == 1
    assert len(archive.impact_methods) == 1
