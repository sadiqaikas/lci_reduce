"""Archive writers for reduced JSON-LD and EcoSpold1 outputs.

These writers are intentionally conservative.  They do not rebuild a database
from scratch; they copy the original archive entries and replace only the
process payloads that the reducer has explicitly filtered.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from .models import DatasetBundle


def write_jsonld_reduced_zip(bundle: DatasetBundle, output_zip: str | Path, process_updates: dict[str, dict]) -> str:
    """Write a new JSON-LD ZIP while replacing only the changed process files.

    Every untouched entry is written back byte-for-byte from the original
    archive snapshot stored in ``bundle.entries``.  This makes preservation
    behaviour easy to reason about during audit.
    """
    target = Path(output_zip)
    with ZipFile(target, "w", compression=ZIP_DEFLATED) as archive:
        for entry in bundle.entries:
            if entry.object_type == "process" and entry.path in process_updates:
                archive.writestr(entry.path, json.dumps(process_updates[entry.path], indent=2, sort_keys=True))
            else:
                archive.writestr(entry.path, entry.raw_bytes)
    return str(target)


def write_ecospold1_reduced_zip(bundle: DatasetBundle, output_zip: str | Path, keep_indices: dict[str, set[int]]) -> str:
    """Write a reduced EcoSpold1 ZIP by filtering exchange elements in-place.

    For EcoSpold1 the reducer rewrites only the XML elements representing
    removed exchange positions.  Other XML content is preserved as parsed and
    reserialised.
    """
    target = Path(output_zip)
    with ZipFile(target, "w", compression=ZIP_DEFLATED) as archive:
        for entry in bundle.entries:
            if entry.kind != "xml" or entry.path not in keep_indices:
                archive.writestr(entry.path, entry.raw_bytes)
                continue
            root = ET.fromstring(entry.raw_bytes.lstrip())
            dataset = next((child for child in list(root) if child.tag.rsplit("}", 1)[-1] == "dataset"), root)
            flow_data = next((child for child in list(dataset) if child.tag.rsplit("}", 1)[-1] == "flowData"), None)
            if flow_data is not None:
                keep = keep_indices[entry.path]
                exchanges = [child for child in list(flow_data) if child.tag.rsplit("}", 1)[-1] == "exchange"]
                for index, child in enumerate(exchanges):
                    if index not in keep:
                        flow_data.remove(child)
            archive.writestr(entry.path, ET.tostring(root, encoding="utf-8", xml_declaration=True))
    return str(target)
