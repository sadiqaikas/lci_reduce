"""EcoSpold1 process and method reader.

EcoSpold1 is structurally very different from openLCA JSON-LD:

- XML instead of JSON;
- process and method exports are shaped differently;
- stable UUID-style identifiers are often absent from the places we need them.

The reader therefore does more identity reconstruction than the JSON-LD reader.
Its goal is still the same: emit the same format-neutral ``DatasetBundle`` so
all downstream reduction and priority mathematics can remain unchanged.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import defaultdict
from uuid import NAMESPACE_URL, uuid5

from .errors import DataFormatError
from .models import ArchiveEntry, DatasetBundle, Exchange, FlowDefinition, ImpactCategory, ImpactFactorCandidate, Location, ProcessRecord, UnitDefinition
from .utils import split_category_path


def _stable_id(kind: str, *parts: str) -> str:
    """Build deterministic synthetic IDs for EcoSpold1 objects.

    EcoSpold1 exports often lack stable UUIDs for the entities we need to match,
    so the clean package derives reproducible IDs from scientifically relevant
    identity fields.

    The important property is reproducibility, not global truth.  As long as
    the same source data yields the same internal IDs, exchanges and factors can
    be matched consistently across repeated runs.
    """
    return str(uuid5(NAMESPACE_URL, "lca_flowkit|" + "|".join(part.strip() for part in parts)))


def _local_name(tag: str) -> str:
    """Drop XML namespace prefixes when matching EcoSpold1 element names.

    XML namespaces are useful for correctness but awkward for everyday parsing.
    The package strips them at comparison time so the code can ask for
    ``"dataset"`` or ``"exchange"`` without repeating namespace boilerplate.
    """
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _children(element: ET.Element, name: str | None = None) -> list[ET.Element]:
    """Collect direct XML children, optionally filtering by local tag name.

    This helper is intentionally simple and only looks one level deep.  The
    calling code remains explicit about the XML structure it expects.
    """
    return [child for child in list(element) if name is None or _local_name(child.tag) == name]


def _first(element: ET.Element | None, name: str) -> ET.Element | None:
    """Return the first matching child element, if any.

    Returning ``None`` instead of raising keeps missing optional EcoSpold1
    fields easy to handle explicitly at the call site.
    """
    if element is None:
        return None
    for child in _children(element, name):
        return child
    return None


def _parse_root(raw_bytes: bytes, path: str) -> ET.Element:
    """Parse one EcoSpold1 XML document and wrap parse errors clearly.

    The error is re-raised as ``DataFormatError`` so workflow code does not
    need XML-library-specific exception handling.
    """
    try:
        return ET.fromstring(raw_bytes.lstrip())
    except ET.ParseError as exc:
        raise DataFormatError(f"Failed to parse EcoSpold1 XML file {path}: {exc}") from exc


def _flow_type(input_group: str, output_group: str) -> str:
    """Map EcoSpold exchange-group codes onto internal flow families.

    The mapping is deliberately narrow because the reducer only needs to know
    whether an exchange is elementary, product, or waste for preservation
    policy.
    """
    if input_group == "4" or output_group == "4":
        return "ELEMENTARY_FLOW"
    if output_group == "3":
        return "WASTE_FLOW"
    return "PRODUCT_FLOW"


def _dataset_is_impact_method(root: ET.Element, dataset: ET.Element) -> bool:
    """Classify one EcoSpold1 dataset as an impact-method result when explicit.

    Namespace information is the strongest signal, but some exports also carry
    the ``impactAssessmentResult`` hint or omit exchange-group codes entirely.
    """
    namespace = root.tag[1:].split("}", 1)[0] if root.tag.startswith("{") else ""
    if "Impact" in namespace:
        return True
    meta = _first(dataset, "metaInformation")
    process_info = _first(meta, "processInformation")
    dataset_info = _first(process_info, "dataSetInformation")
    if dataset_info is not None and str(dataset_info.get("impactAssessmentResult") or "").strip().casefold() in {"1", "true", "yes"}:
        return True
    reference = _first(process_info, "referenceFunction")
    if reference is None:
        return False
    if str(reference.get("datasetRelatesToProduct") or "").strip().casefold() != "false":
        return False
    flow_data = _first(dataset, "flowData")
    if flow_data is None:
        return False
    exchanges = _children(flow_data, "exchange")
    if not exchanges:
        return False
    has_exchange_groups = any(_first(exchange, "inputGroup") is not None or _first(exchange, "outputGroup") is not None for exchange in exchanges)
    return not has_exchange_groups


def _dataset_is_process(root: ET.Element, dataset: ET.Element) -> bool:
    """Classify one EcoSpold1 dataset as a process/activity inventory.

    The reader requires the semantic markers that the workflows actually need:
    process metadata, a reference function, and flow data.  Folder names or the
    mere fact that a file is XML are not enough.
    """
    if _dataset_is_impact_method(root, dataset):
        return False
    meta = _first(dataset, "metaInformation")
    process_info = _first(meta, "processInformation")
    reference = _first(process_info, "referenceFunction")
    flow_data = _first(dataset, "flowData")
    return reference is not None and flow_data is not None


def load_ecospold1_bundle(source: str, entries_raw: list[tuple[str, bytes]]) -> DatasetBundle:
    """Load EcoSpold1 processes and methods into the neutral data model.

    The function accepts a mixed folder or archive of EcoSpold1 XML files and
    classifies each file as either:

    - a process inventory export; or
    - an impact-method export.

    Every parsed file is still preserved in ``entries`` so a reduced ZIP can be
    written later without re-reading the original source.
    """
    entries: list[ArchiveEntry] = []
    flows: dict[str, FlowDefinition] = {}
    units: dict[str, UnitDefinition] = {}
    categories: dict[str, ImpactCategory] = {}
    processes: list[ProcessRecord] = []
    methods: dict[str, dict[str, str]] = {}
    parse_warnings: list[dict[str, str]] = []

    for path, raw_bytes in entries_raw:
        if not path.lower().endswith((".xml", ".spold")):
            entries.append(ArchiveEntry(path=path, raw_bytes=raw_bytes, kind="binary", object_type="other", object_id=path, name=path))
            continue
        root = _parse_root(raw_bytes, path)
        entries.append(ArchiveEntry(path=path, raw_bytes=raw_bytes, kind="xml", object_type="dataset", object_id=path, name=path, parsed=root))
        dataset = _first(root, "dataset") if _local_name(root.tag) == "ecoSpold" else root
        if dataset is None:
            continue
        meta = _first(dataset, "metaInformation")
        process_info = _first(meta, "processInformation")
        reference = _first(process_info, "referenceFunction")
        if reference is None:
            parse_warnings.append({"source_file": path, "message": "Missing processInformation/referenceFunction"})
            continue
        if _dataset_is_impact_method(root, dataset):
            method_name = str(reference.get("category") or "").strip()
            category_name = str(reference.get("name") or "").strip()
            method_id = _stable_id("method", method_name)
            category_id = _stable_id("category", method_name, category_name)
            methods[method_id] = {"name": method_name, "source_path": path}
            category = ImpactCategory(
                category_id=category_id,
                name=category_name,
                method_id=method_id,
                method_name=method_name,
                reference_unit=str(reference.get("unit") or "").strip(),
                path=f"{method_name}/{str(reference.get('subCategory') or '').strip()}".strip("/"),
                source_path=path,
                method_source_path=path,
            )
            flow_data = _first(dataset, "flowData")
            for exchange in _children(flow_data, "exchange") if flow_data is not None else []:
                flow_name = str(exchange.get("name") or "").strip()
                category_path = "/".join(part for part in (exchange.get("category"), exchange.get("subCategory")) if part)
                flow_id = _stable_id("flow", flow_name, category_path, "ELEMENTARY_FLOW")
                compartment, subcompartment = split_category_path(category_path)
                category.factors_by_flow.setdefault(flow_id, []).append(
                    ImpactFactorCandidate(
                        category_id=category.category_id,
                        category_name=category.name,
                        method_id=method_id,
                        method_name=method_name,
                        flow_id=flow_id,
                        flow_name=flow_name,
                        value=float(exchange.get("meanValue") or 0.0),
                        unit_name=str(exchange.get("unit") or "").strip(),
                        compartment=compartment,
                        subcompartment=subcompartment,
                        source_path=path,
                        raw=dict(exchange.attrib),
                    )
                )
            categories[category_id] = category
            continue
        if not _dataset_is_process(root, dataset):
            parse_warnings.append(
                {
                    "source_file": path,
                    "message": "Skipped non-process EcoSpold1 dataset that was neither a process/activity inventory nor an impact-method dataset",
                }
            )
            continue

        process_name = str(reference.get("name") or "").strip() or path
        geography = _first(process_info, "geography")
        process_location = Location(name=str(geography.get("location") or "").strip(), region=str(geography.get("location") or "").strip()) if geography is not None else Location()
        process_id = _stable_id("process", path, process_name, process_location.name)
        flow_data = _first(dataset, "flowData")
        exchange_rows: list[Exchange] = []
        for index, exchange in enumerate(_children(flow_data, "exchange") if flow_data is not None else []):
            input_group = (_first(exchange, "inputGroup").text or "").strip() if _first(exchange, "inputGroup") is not None else ""
            output_group = (_first(exchange, "outputGroup").text or "").strip() if _first(exchange, "outputGroup") is not None else ""
            flow_name = str(exchange.get("name") or "").strip()
            category_path = "/".join(part for part in (exchange.get("category"), exchange.get("subCategory")) if part)
            flow_type = _flow_type(input_group, output_group)
            # The synthetic flow ID must be built from the same identity fields
            # that the method reader uses, otherwise process exchanges and LCIA
            # factors for the same elementary flow would never meet.
            flow_id = _stable_id("flow", flow_name, category_path, flow_type)
            unit_name = str(exchange.get("unit") or "").strip()
            if unit_name:
                unit_id = _stable_id("unit", unit_name)
                units.setdefault(unit_id, UnitDefinition(unit_id=unit_id, name=unit_name))
            else:
                unit_id = ""
            if flow_id not in flows:
                compartment, subcompartment = split_category_path(category_path)
                flows[flow_id] = FlowDefinition(
                    flow_id=flow_id,
                    name=flow_name,
                    flow_type=flow_type,
                    compartment=compartment,
                    subcompartment=subcompartment,
                    is_elementary=flow_type == "ELEMENTARY_FLOW",
                )
            exchange_rows.append(
                Exchange(
                    exchange_id=_stable_id("exchange", process_id, str(index), flow_id),
                    index=index,
                    amount=float(exchange.get("meanValue") or 0.0),
                    flow_id=flow_id,
                    flow_name=flow_name,
                    unit_name=unit_name,
                    unit_id=unit_id,
                    provider_id="",
                    is_quantitative_reference=output_group == "0",
                    raw=exchange,
                )
            )
        processes.append(
            ProcessRecord(
                process_id=process_id,
                name=process_name,
                source_path=path,
                location=process_location,
                exchanges=exchange_rows,
                raw=root,
            )
        )

    return DatasetBundle(
        source_path=source,
        source_name=source.rsplit("/", 1)[-1],
        source_format="ecospold1",
        entries=entries,
        processes=processes,
        flows=flows,
        units=units,
        categories=categories,
        methods=methods,
        extra={"parse_warnings": parse_warnings},
    )
