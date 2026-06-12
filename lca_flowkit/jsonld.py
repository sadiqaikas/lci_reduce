"""openLCA JSON-LD reader.

This module is intentionally verbose in its parsing choices because the reader
defines the identities that later scientific steps rely on.

The boundary principle is:

- be tolerant about where openLCA-style exports store a field;
- be strict about the internal model produced from that field.

Once parsing is complete, the rest of the package works only with
format-neutral dataclasses such as ``FlowDefinition``, ``Exchange``, and
``ImpactFactorCandidate``.  That separation keeps the mathematical code free of
file-format details.
"""

from __future__ import annotations

from typing import Any

from .models import ArchiveEntry, DatasetBundle, Exchange, FlowDefinition, ImpactCategory, ImpactFactorCandidate, Location, ProcessRecord, UnitDefinition
from .utils import location_from_any, object_id, object_name, parse_json_object, reference_id, split_category_path


_TOP_LEVEL_OBJECT_TYPES = {
    "processes": "process",
    "flows": "flow",
    "flow_properties": "flow_property",
    "flowproperties": "flow_property",
    "categories": "category",
    "category": "category",
    "flow_categories": "category",
    "flowcategories": "category",
    "units": "unit",
    "unit_groups": "unit_group",
    "unitgroups": "unit_group",
    "lcia_methods": "impact_method",
    "impact_methods": "impact_method",
    "lcia_categories": "impact_category",
    "impact_categories": "impact_category",
}
_NON_DATASET_TOP_LEVEL_SEGMENTS = {"bin"}
_RECOGNISED_COLLECTION_SEGMENTS = set(_TOP_LEVEL_OBJECT_TYPES) | _NON_DATASET_TOP_LEVEL_SEGMENTS


def _normalise_type_token(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum() or ch == "_")


def _path_segments(path: str) -> list[str]:
    parts = [part.strip() for part in path.replace("\\", "/").split("/") if part.strip()]
    return [part.casefold() for part in parts]


def _collection_segment(path: str) -> str:
    parts = _path_segments(path)
    if not parts:
        return ""
    if parts[0] in _RECOGNISED_COLLECTION_SEGMENTS:
        return parts[0]
    if len(parts) >= 2 and parts[1] in _RECOGNISED_COLLECTION_SEGMENTS:
        return parts[1]
    return parts[0]


def _explicit_object_type(data: dict[str, Any]) -> str:
    for candidate in (data.get("@type"), data.get("type"), data.get("olcaType")):
        if not isinstance(candidate, str):
            continue
        token = _normalise_type_token(candidate)
        mapping = {
            "process": "process",
            "flow": "flow",
            "flowproperty": "flow_property",
            "flow_property": "flow_property",
            "impactcategory": "impact_category",
            "impactmethod": "impact_method",
            "unit": "unit",
            "unitgroup": "unit_group",
            "unit_group": "unit_group",
            "category": "category",
        }
        if token in mapping:
            return mapping[token]
    return ""


def _object_type(data: dict[str, Any], path: str) -> str:
    """Infer the object type from explicit metadata plus top-level collection.

    The reader deliberately avoids broad substring matching such as
    ``"process" in path`` because openLCA archives can contain sidecar files
    under lookalike paths like ``bin/processes/...`` that are not reducible
    datasets.
    """
    explicit_type = _explicit_object_type(data)
    collection = _collection_segment(path)
    path_type = _TOP_LEVEL_OBJECT_TYPES.get(collection, "")
    if explicit_type:
        if collection in _NON_DATASET_TOP_LEVEL_SEGMENTS:
            return "other"
        if path_type and path_type != explicit_type:
            return "other"
        return explicit_type
    if path_type:
        return path_type
    return "other"


def _category_lookup(entries: list[ArchiveEntry]) -> dict[str, str]:
    """Resolve category IDs to full category paths for flow classification.

    Flow matching later uses compartment and subcompartment text.  Many JSON-LD
    objects only store a category reference, so this helper reconstructs the
    readable category path by following parent links recursively.
    """
    category_entries = {entry.object_id: entry for entry in entries if entry.object_type == "category"}
    cache: dict[str, str] = {}

    def resolve(category_id: str) -> str:
        if not category_id:
            return ""
        if category_id in cache:
            return cache[category_id]
        entry = category_entries.get(category_id)
        if entry is None or not isinstance(entry.parsed, dict):
            return ""
        parent_id = reference_id(entry.parsed.get("parentCategory") or entry.parsed.get("category") or entry.parsed.get("parent"))
        parent_path = resolve(parent_id)
        name = entry.name
        path = str(entry.parsed.get("categoryPath") or entry.parsed.get("path") or "").strip()
        if path and name and not path.endswith(name):
            path = f"{path.rstrip('/')}/{name}"
        if not path:
            path = f"{parent_path}/{name}".strip("/")
        cache[category_id] = path
        return path

    for category_id in list(category_entries):
        resolve(category_id)
    return cache


def _extract_units(entries: list[ArchiveEntry]) -> dict[str, UnitDefinition]:
    """Extract units from both standalone unit objects and unit groups.

    openLCA can encode units either as top-level ``Unit`` objects or nested
    inside ``UnitGroup`` objects.  The package normalises both styles into one
    registry so later unit resolution does not need to care how the source was
    structured.
    """
    units: dict[str, UnitDefinition] = {}
    for entry in entries:
        data = entry.parsed
        if not isinstance(data, dict):
            continue
        if entry.object_type == "unit":
            units[entry.object_id] = UnitDefinition(
                unit_id=entry.object_id,
                name=entry.name,
                group_id=reference_id(data.get("unitGroup")),
                group_name=object_name(data.get("unitGroup")),
                conversion_factor=float(data["conversionFactor"]) if data.get("conversionFactor") is not None else None,
                is_reference_unit=bool(data.get("referenceUnit") or data.get("isReferenceUnit")),
                flow_property_id=reference_id(data.get("defaultFlowProperty") or data.get("referenceFlowProperty") or data.get("flowProperty")),
                flow_property_name=object_name(data.get("defaultFlowProperty") or data.get("referenceFlowProperty") or data.get("flowProperty")),
                raw=data,
            )
        elif entry.object_type == "unit_group":
            ref_unit_id = reference_id(data.get("referenceUnit"))
            flow_property_id = reference_id(data.get("defaultFlowProperty") or data.get("referenceFlowProperty") or data.get("flowProperty"))
            flow_property_name = object_name(data.get("defaultFlowProperty") or data.get("referenceFlowProperty") or data.get("flowProperty"))
            for unit_data in data.get("units", []) or []:
                if not isinstance(unit_data, dict):
                    continue
                unit_id = object_id(unit_data)
                if not unit_id:
                    continue
                factor = unit_data.get("conversionFactor")
                if factor is None and ref_unit_id and unit_id == ref_unit_id:
                    factor = 1.0
                units[unit_id] = UnitDefinition(
                    unit_id=unit_id,
                    name=str(unit_data.get("name") or ""),
                    group_id=entry.object_id,
                    group_name=entry.name,
                    conversion_factor=float(factor) if factor is not None else None,
                    is_reference_unit=bool(unit_data.get("referenceUnit") or unit_data.get("isReferenceUnit") or unit_id == ref_unit_id),
                    flow_property_id=flow_property_id,
                    flow_property_name=flow_property_name,
                    raw=unit_data,
                )
    return units


def _extract_flows(entries: list[ArchiveEntry], category_paths: dict[str, str]) -> dict[str, FlowDefinition]:
    """Build the flow registry used during contribution and reduction steps.

    The flow registry is where format-specific flow metadata becomes the stable
    internal identity used for:

    - matching exchanges to factors by flow ID;
    - checking compartment/subcompartment compatibility; and
    - determining whether a flow is elementary and thus reducible.
    """
    flows: dict[str, FlowDefinition] = {}
    for entry in entries:
        data = entry.parsed
        if entry.object_type != "flow" or not isinstance(data, dict):
            continue
        flow_type = str(data.get("flowType") or "").strip() or "FLOW"
        category_path = str(data.get("categoryPath") or data.get("path") or "").strip()
        if not category_path:
            category_path = category_paths.get(reference_id(data.get("category")), "")
        compartment, subcompartment = split_category_path(category_path)
        ref_prop = data.get("referenceFlowProperty")
        if not isinstance(ref_prop, dict):
            ref_prop = {}
        for item in data.get("flowProperties", []) or []:
            if isinstance(item, dict) and (
                item.get("referenceFlowProperty") or item.get("isReferenceFlowProperty") or item.get("reference")
            ):
                prop = item.get("flowProperty")
                if isinstance(prop, dict):
                    ref_prop = prop
                    break
        flows[entry.object_id] = FlowDefinition(
            flow_id=entry.object_id,
            name=entry.name,
            flow_type=flow_type,
            compartment=compartment,
            subcompartment=subcompartment,
            is_elementary=flow_type == "ELEMENTARY_FLOW",
            location=location_from_any(data.get("location")),
            reference_flow_property_id=reference_id(ref_prop),
            reference_flow_property_name=object_name(ref_prop),
            raw=data,
        )
    return flows


def _extract_processes(entries: list[ArchiveEntry]) -> list[ProcessRecord]:
    """Extract process datasets and convert exchange payloads to typed rows.

    The reducer eventually needs to remove exchanges by *position* from the
    original process JSON.  For that reason every ``Exchange`` stores both a
    stable exchange ID and the original list index.
    """
    processes: list[ProcessRecord] = []
    for entry in entries:
        data = entry.parsed
        if entry.object_type != "process" or not isinstance(data, dict):
            continue
        exchanges: list[Exchange] = []
        for index, exchange_data in enumerate(data.get("exchanges", []) or []):
            if not isinstance(exchange_data, dict):
                continue
            unit_payload = exchange_data.get("unit") or exchange_data.get("referenceUnit") or {}
            flow_property = exchange_data.get("flowProperty") or {}
            exchanges.append(
                Exchange(
                    exchange_id=object_id(exchange_data, fallback=f"{entry.object_id}:exchange:{index}"),
                    index=index,
                    amount=float(exchange_data.get("amount") or 0.0),
                    flow_id=reference_id(exchange_data.get("flow")),
                    flow_name=object_name(exchange_data.get("flow")),
                    unit_name=object_name(unit_payload) or str(exchange_data.get("unitName") or ""),
                    unit_id=reference_id(unit_payload),
                    flow_property_id=reference_id(flow_property),
                    flow_property_name=object_name(flow_property),
                    location=location_from_any(exchange_data.get("location")),
                    provider_id=reference_id(exchange_data.get("provider")),
                    is_quantitative_reference=bool(
                        exchange_data.get("quantitativeReference")
                        or exchange_data.get("isQuantitativeReference")
                        or exchange_data.get("referenceFlow")
                    ),
                    raw=exchange_data,
                )
            )
        processes.append(
            ProcessRecord(
                process_id=entry.object_id,
                name=entry.name,
                source_path=entry.path,
                location=location_from_any(data.get("location")),
                exchanges=exchanges,
                raw=data,
            )
        )
    return processes


def _extract_categories(entries: list[ArchiveEntry], method_lookup: dict[str, dict[str, str]]) -> dict[str, ImpactCategory]:
    """Extract LCIA categories and preserve all admissible factor candidates.

    The reader does not collapse factor ambiguity.  If the source contains
    multiple admissible factors for the same flow within one category, every
    candidate is preserved so the scenario builder can make that ambiguity
    explicit later.
    """
    categories: dict[str, ImpactCategory] = {}
    for entry in entries:
        data = entry.parsed
        if entry.object_type != "impact_category" or not isinstance(data, dict):
            continue
        method_ref = data.get("impactMethod") or data.get("method")
        method_id = reference_id(method_ref)
        method_meta = method_lookup.get(method_id, {})
        category = ImpactCategory(
            category_id=entry.object_id,
            name=entry.name,
            method_id=method_id,
            method_name=object_name(method_ref) or method_meta.get("name", ""),
            reference_unit=str(data.get("referenceUnitName") or data.get("refUnitName") or data.get("refUnit") or ""),
            path=str(data.get("categoryPath") or ""),
            source_path=entry.path,
            method_source_path=method_meta.get("source_path", ""),
        )
        for factor_data in data.get("impactFactors", []) or []:
            if not isinstance(factor_data, dict):
                continue
            flow_id = reference_id(factor_data.get("flow"))
            if not flow_id:
                continue
            location = location_from_any(factor_data.get("location"))
            category.factors_by_flow.setdefault(flow_id, []).append(
                ImpactFactorCandidate(
                    category_id=category.category_id,
                    category_name=category.name,
                    method_id=category.method_id,
                    method_name=category.method_name,
                    flow_id=flow_id,
                    flow_name=object_name(factor_data.get("flow")),
                    value=float(factor_data.get("value") or 0.0),
                    unit_name=object_name(factor_data.get("unit"))
                    or str(factor_data.get("unitName") or factor_data.get("referenceUnitName") or category.reference_unit),
                    unit_id=reference_id(factor_data.get("unit")),
                    flow_property_id=reference_id(
                        factor_data.get("flowProperty") or factor_data.get("flowPropertyRef") or factor_data.get("referenceFlowProperty")
                    ),
                    flow_property_name=object_name(
                        factor_data.get("flowProperty") or factor_data.get("flowPropertyRef") or factor_data.get("referenceFlowProperty")
                    ),
                    location=location,
                    compartment=str(factor_data.get("compartment") or ""),
                    subcompartment=str(factor_data.get("subCompartment") or factor_data.get("subcompartment") or ""),
                    source_path=entry.path,
                    raw=factor_data,
                )
            )
        categories[category.category_id] = category
    return categories


def load_jsonld_bundle(source: str, entries_raw: list[tuple[str, bytes]]) -> DatasetBundle:
    """Load one JSON-LD source bundle into the internal neutral data model.

    The function keeps every archive entry, even files that are not directly
    used by the mathematical workflows.  That allows the writer to reproduce a
    reduced ZIP that preserves untouched database objects byte-for-byte.
    """
    entries: list[ArchiveEntry] = []
    for path, raw_bytes in entries_raw:
        if not path.lower().endswith((".json", ".jsonld")):
            entries.append(ArchiveEntry(path=path, raw_bytes=raw_bytes, kind="binary", object_type="other", object_id=path, name=path))
            continue
        data = parse_json_object(raw_bytes, path)
        entry = ArchiveEntry(
            path=path,
            raw_bytes=raw_bytes,
            kind="json",
            object_type=_object_type(data, path),
            object_id=object_id(data, fallback=path),
            name=object_name(data) or path,
            parsed=data,
        )
        entries.append(entry)
    category_paths = _category_lookup(entries)
    method_lookup = {
        entry.object_id: {"name": entry.name, "source_path": entry.path}
        for entry in entries
        if entry.object_type == "impact_method"
    }
    return DatasetBundle(
        source_path=source,
        source_name=source.rsplit("/", 1)[-1],
        source_format="jsonld",
        entries=entries,
        processes=_extract_processes(entries),
        flows=_extract_flows(entries, category_paths),
        units=_extract_units(entries),
        categories=_extract_categories(entries, method_lookup),
        methods=method_lookup,
    )
