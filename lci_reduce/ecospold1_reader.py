"""EcoSpold1 archive parsing and writing helpers."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple
from uuid import NAMESPACE_URL, uuid5
from zipfile import ZIP_DEFLATED, BadZipFile, ZipFile

from .errors import DataFormatError
from .models import (
    CategoryPathDiagnostics,
    CharacterisationFactorCandidate,
    CharacterizationFactor,
    DatasetEntry,
    DatasetLocator,
    FlowInfo,
    ImpactCategory,
    JsonLdArchive,
    JsonLdArchiveIndex,
    ProcessReductionResult,
    UnitInfo,
)


ECOSPOLD_SUFFIXES = {".xml", ".spold", ".zip"}
ECOSPOLD_ROOT_TAG = "ecoSpold"
XSI_NAMESPACE = "http://www.w3.org/2001/XMLSchema-instance"
ECOSPOLD_XML_ERROR_POLICIES = {"fail", "warn_skip"}


def _normalise_text(value: str | None) -> str:
    if value is None:
        return ""
    return " ".join(value.strip().split())


def _normalise_key(value: str | None) -> str:
    return _normalise_text(value).lower()


def _stable_id(kind: str, *parts: str) -> str:
    token = "|".join([kind, *(_normalise_text(part) for part in parts)])
    return str(uuid5(NAMESPACE_URL, f"lci_reduce:ecospold1:{token}"))


def _iter_source_entries(source: str) -> Iterator[tuple[str, bytes]]:
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(source)
    if path.is_dir():
        for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
            yield file_path.relative_to(path).as_posix(), file_path.read_bytes()
        return
    suffix = path.suffix.lower()
    if suffix == ".zip":
        try:
            with ZipFile(path, "r") as archive:
                for name in sorted(archive.namelist()):
                    if name.endswith("/"):
                        continue
                    yield name, archive.read(name)
        except BadZipFile as exc:
            raise DataFormatError(f"Failed to open archive {source}: {exc}") from exc
        return
    if suffix in {".xml", ".spold"}:
        yield path.name, path.read_bytes()
        return
    raise DataFormatError(
        f"Unsupported EcoSpold1 input source: {source}. Supported inputs are folders, .zip archives, .xml files, and .spold files."
    )


def _local_name(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[1]
    return tag


def _namespace_uri(tag: str) -> str:
    if tag.startswith("{") and "}" in tag:
        return tag[1:].split("}", 1)[0]
    return ""


def _iter_children(element: ET.Element, name: str | None = None) -> Iterator[ET.Element]:
    for child in list(element):
        if name is None or _local_name(child.tag) == name:
            yield child


def _find_child(element: ET.Element | None, name: str) -> ET.Element | None:
    if element is None:
        return None
    for child in _iter_children(element, name):
        return child
    return None


def _child_text(element: ET.Element | None, name: str) -> str:
    child = _find_child(element, name)
    if child is None or child.text is None:
        return ""
    return child.text.strip()


def _bool_attr(element: ET.Element | None, name: str) -> bool:
    if element is None:
        return False
    value = _normalise_key(element.get(name))
    return value in {"1", "true", "yes"}


def _parse_root(raw_bytes: bytes, rel_path: str) -> ET.Element:
    try:
        return ET.fromstring(raw_bytes.lstrip())
    except ET.ParseError as exc:
        raise DataFormatError(f"Failed to parse EcoSpold1 XML file {rel_path}: {exc}") from exc


def _try_parse_ecospold_root(raw_bytes: bytes, rel_path: str) -> ET.Element | None:
    root = _parse_root(raw_bytes, rel_path)
    return root if _local_name(root.tag) == ECOSPOLD_ROOT_TAG else None


def _dataset_key(rel_path: str, dataset_index: int) -> str:
    return f"{rel_path}::dataset:{dataset_index}"


def _category_path(category: str, subcategory: str) -> str:
    parts = [part.strip() for part in (category, subcategory) if part and part.strip()]
    return "/".join(parts)


def _parse_unit_id(unit_name: str) -> str:
    return _stable_id("unit", unit_name or "-")


def _parse_unit_registry(unit_names: Iterable[str]) -> Dict[str, UnitInfo]:
    units: Dict[str, UnitInfo] = {}
    for unit_name in sorted({_normalise_text(value) for value in unit_names if _normalise_text(value)}):
        unit_id = _parse_unit_id(unit_name)
        units[unit_id] = UnitInfo(
            unit_id=unit_id,
            name=unit_name,
            group_id=None,
            raw={"name": unit_name},
            group_name=None,
            conversion_factor=None,
            is_reference_unit=False,
            flow_property_id=None,
            flow_property_name=None,
        )
    return units


def _parse_exchange_groups(exchange: ET.Element) -> tuple[str, str]:
    return _child_text(exchange, "inputGroup"), _child_text(exchange, "outputGroup")


def _is_elementary_exchange(input_group: str, output_group: str) -> bool:
    return input_group == "4" or output_group == "4"


def _flow_type(input_group: str, output_group: str) -> str:
    if _is_elementary_exchange(input_group, output_group):
        return "ELEMENTARY_FLOW"
    if output_group == "3":
        return "WASTE_FLOW"
    return "PRODUCT_FLOW"


def _flow_key(name: str, category: str, subcategory: str, *, elementary: bool) -> str:
    if elementary:
        return _stable_id("elementary-flow", name, category, subcategory)
    return _stable_id("intermediate-flow", name, category, subcategory)


def _location_payload(location_code: str) -> dict[str, str] | None:
    token = _normalise_text(location_code)
    if not token:
        return None
    return {
        "@id": _stable_id("location", token),
        "name": token,
        "code": token,
        "region": token,
    }


def _flow_info_from_exchange(
    flow_id: str,
    *,
    name: str,
    category: str,
    subcategory: str,
    flow_type: str,
    source_file: str = "",
) -> FlowInfo:
    category_path = _category_path(category, subcategory)
    return FlowInfo(
        flow_id=flow_id,
        name=name,
        flow_type=flow_type,
        category_path=category_path,
        is_elementary=flow_type == "ELEMENTARY_FLOW",
        raw={
            "@id": flow_id,
            "name": name,
            "flowType": flow_type,
            "categoryPath": category_path,
            "flowProperties": [],
        },
        source_file=source_file,
    )


def _parse_amount(exchange: ET.Element, rel_path: str) -> float:
    raw_value = exchange.get("meanValue")
    if raw_value is None:
        raise DataFormatError(f"EcoSpold1 exchange in {rel_path} is missing meanValue")
    try:
        return float(raw_value)
    except ValueError as exc:
        raise DataFormatError(f"EcoSpold1 exchange in {rel_path} has invalid meanValue={raw_value!r}") from exc


def _parse_process_dataset(
    dataset: ET.Element,
    *,
    rel_path: str,
    dataset_index: int,
    flows: Dict[str, FlowInfo],
    unit_names: set[str],
) -> tuple[DatasetLocator, DatasetEntry]:
    meta = _find_child(dataset, "metaInformation")
    process_info = _find_child(meta, "processInformation")
    reference_function = _find_child(process_info, "referenceFunction")
    geography = _find_child(process_info, "geography")
    if reference_function is None or process_info is None:
        raise DataFormatError(f"EcoSpold1 process dataset {rel_path} is missing processInformation/referenceFunction")

    process_name = _normalise_text(reference_function.get("name")) or f"Dataset {dataset_index + 1}"
    dataset_number = _normalise_text(dataset.get("number")) or str(dataset_index + 1)
    location_code = _normalise_text(geography.get("location") if geography is not None else "")
    process_id = _stable_id("process", rel_path, str(dataset_index), dataset_number, process_name, location_code)
    flow_data = _find_child(dataset, "flowData")
    if flow_data is None:
        raise DataFormatError(f"EcoSpold1 process dataset {rel_path} is missing flowData")

    exchanges: List[dict[str, Any]] = []
    for exchange_index, exchange in enumerate(_iter_children(flow_data, "exchange")):
        input_group, output_group = _parse_exchange_groups(exchange)
        name = _normalise_text(exchange.get("name")) or f"Exchange {exchange_index + 1}"
        category = _normalise_text(exchange.get("category"))
        subcategory = _normalise_text(exchange.get("subCategory"))
        flow_type = _flow_type(input_group, output_group)
        flow_id = _flow_key(name, category, subcategory, elementary=flow_type == "ELEMENTARY_FLOW")
        flows.setdefault(
            flow_id,
            _flow_info_from_exchange(
                flow_id,
                name=name,
                category=category,
                subcategory=subcategory,
                flow_type=flow_type,
                source_file=rel_path,
            ),
        )
        unit_name = _normalise_text(exchange.get("unit"))
        if unit_name:
            unit_names.add(unit_name)
        unit_payload = {"@id": _parse_unit_id(unit_name), "name": unit_name} if unit_name else None
        exchange_location = _location_payload(_normalise_text(exchange.get("location")))
        exchange_id = _stable_id(
            "exchange",
            process_id,
            _normalise_text(exchange.get("number")) or str(exchange_index),
            flow_id,
        )
        exchange_data: dict[str, Any] = {
            "@id": exchange_id,
            "internalId": _normalise_text(exchange.get("number")) or str(exchange_index + 1),
            "amount": _parse_amount(exchange, rel_path),
            "flow": {"@id": flow_id, "name": name},
            "unitName": unit_name,
            "inputGroup": input_group,
            "outputGroup": output_group,
            "categoryPath": _category_path(category, subcategory),
            "infrastructureProcess": _bool_attr(exchange, "infrastructureProcess"),
        }
        if unit_payload is not None:
            exchange_data["unit"] = unit_payload
        if exchange_location is not None:
            exchange_data["location"] = exchange_location
        if output_group == "0":
            exchange_data["quantitativeReference"] = True
        exchanges.append(exchange_data)

    process_data = {
        "@id": process_id,
        "@type": "Process",
        "name": process_name,
        "location": _location_payload(location_code) or {},
        "exchanges": exchanges,
    }
    key = _dataset_key(rel_path, dataset_index)
    locator = DatasetLocator(
        path=key,
        object_type="process",
        object_id=process_id,
        name=process_name,
    )
    entry = DatasetEntry(
        path=key,
        raw_bytes=b"",
        data=process_data,
        object_type="process",
        object_id=process_id,
        name=process_name,
    )
    return locator, entry


def _impact_method_name(method_family: str, method_branch: str) -> str:
    family_text = _normalise_text(method_family)
    branch_text = _normalise_text(method_branch)
    return family_text or branch_text or "EcoSpold1 method"


def _impact_category_name(reference_function: ET.Element, dataset_index: int) -> str:
    method_branch = _normalise_text(reference_function.get("subCategory"))
    explicit_name = _normalise_text(reference_function.get("name"))
    return explicit_name or method_branch or f"Impact category {dataset_index + 1}"


def _impact_category_from_dataset(
    dataset: ET.Element,
    *,
    rel_path: str,
    dataset_index: int,
    unit_names: set[str],
) -> tuple[str, ImpactCategory, DatasetEntry]:
    meta = _find_child(dataset, "metaInformation")
    process_info = _find_child(meta, "processInformation")
    reference_function = _find_child(process_info, "referenceFunction")
    if reference_function is None:
        raise DataFormatError(f"EcoSpold1 impact dataset {rel_path} is missing processInformation/referenceFunction")

    method_family = _normalise_text(reference_function.get("category"))
    method_branch = _normalise_text(reference_function.get("subCategory"))
    category_name = _impact_category_name(reference_function, dataset_index)
    reference_unit = _normalise_text(reference_function.get("unit"))
    if reference_unit:
        unit_names.add(reference_unit)
    method_id = _stable_id("impact-method", method_family or rel_path)
    category_id = _stable_id("impact-category", method_id, category_name)
    method_name = _impact_method_name(method_family, method_branch)
    category_path = _category_path(method_family, category_name)
    method_path = _category_path(method_family, "")
    flow_data = _find_child(dataset, "flowData")
    if flow_data is None:
        raise DataFormatError(f"EcoSpold1 impact dataset {rel_path} is missing flowData")

    factor_candidates: Dict[str, List[CharacterisationFactorCandidate]] = defaultdict(list)
    factors: Dict[str, CharacterizationFactor] = {}
    raw_factors: List[dict[str, Any]] = []
    for factor_index, factor in enumerate(_iter_children(flow_data)):
        if _local_name(factor.tag) != "exchange":
            continue
        flow_name = _normalise_text(factor.get("name")) or f"Flow {factor_index + 1}"
        compartment = _normalise_text(factor.get("category"))
        subcompartment = _normalise_text(factor.get("subCategory"))
        unit_name = _normalise_text(factor.get("unit"))
        if unit_name:
            unit_names.add(unit_name)
        flow_id = _flow_key(flow_name, compartment, subcompartment, elementary=True)
        raw_factor_object = {
            "name": flow_name,
            "category": compartment,
            "subCategory": subcompartment,
            "unit": unit_name,
            "meanValue": factor.get("meanValue"),
            "location": _normalise_text(factor.get("location")),
        }
        candidate = CharacterisationFactorCandidate(
            category_id=category_id,
            category_name=category_name,
            method_id=method_id,
            method_name=method_name,
            flow_id=flow_id,
            flow_name=flow_name,
            cf_value=_parse_amount(factor, rel_path),
            cf_unit=unit_name or None,
            cf_unit_id=_parse_unit_id(unit_name) if unit_name else None,
            cf_flow_property_id=None,
            cf_flow_property_name=None,
            cf_location_id=None,
            cf_location_name=None,
            cf_region=_normalise_text(factor.get("location")) or None,
            cf_compartment=compartment or None,
            cf_subcompartment=subcompartment or None,
            source_file=_dataset_key(rel_path, dataset_index),
            raw_factor_object=raw_factor_object,
        )
        factor_candidates[flow_id].append(candidate)
        raw_factors.append(raw_factor_object)
        if flow_id not in factors:
            factors[flow_id] = CharacterizationFactor(
                flow_id=flow_id,
                value=candidate.cf_value,
                unit_name=candidate.cf_unit,
                raw=raw_factor_object,
            )

    raw = {
        "@id": category_id,
        "@type": "ImpactCategory",
        "name": category_name,
        "impactMethod": {"@id": method_id, "name": method_name},
        "categoryPath": category_path,
        "referenceUnit": {"name": reference_unit} if reference_unit else {},
        "impactFactors": raw_factors,
    }
    category = ImpactCategory(
        category_id=category_id,
        name=category_name,
        method_id=method_id,
        method_name=method_name,
        path=category_path,
        metadata_text=" ".join(part for part in (method_name, category_name, reference_unit, category_path) if part),
        reference_unit=reference_unit or None,
        factors=factors,
        raw=raw,
        factor_candidates=dict(factor_candidates),
        source_file=_dataset_key(rel_path, dataset_index),
        method_path=method_path,
        method_source_file=rel_path,
    )
    method_entry = DatasetEntry(
        path=_dataset_key(rel_path, dataset_index),
        raw_bytes=b"",
        data={
            "@id": method_id,
            "@type": "ImpactMethod",
            "name": method_name,
            "categoryPath": method_path,
        },
        object_type="impact_method",
        object_id=method_id,
        name=method_name,
    )
    return method_id, category, method_entry


def _dataset_is_impact_category(dataset: ET.Element, *, root_namespace: str = "") -> bool:
    namespace_token = _normalise_key(root_namespace)
    if namespace_token.endswith("ecospold01impact"):
        return True
    meta = _find_child(dataset, "metaInformation")
    process_info = _find_child(meta, "processInformation")
    dataset_info = _find_child(process_info, "dataSetInformation")
    if _bool_attr(dataset_info, "impactAssessmentResult"):
        return True
    reference_function = _find_child(process_info, "referenceFunction")
    if reference_function is None:
        return False
    if _normalise_key(reference_function.get("datasetRelatesToProduct")) != "false":
        return False
    flow_data = _find_child(dataset, "flowData")
    if flow_data is None:
        return False
    exchanges = list(_iter_children(flow_data, "exchange"))
    if not exchanges:
        return False
    has_exchange_groups = any(_child_text(exchange, "inputGroup") or _child_text(exchange, "outputGroup") for exchange in exchanges)
    return not has_exchange_groups


def _dataset_is_process_dataset(dataset: ET.Element, *, root_namespace: str = "") -> bool:
    if _dataset_is_impact_category(dataset, root_namespace=root_namespace):
        return False
    meta = _find_child(dataset, "metaInformation")
    process_info = _find_child(meta, "processInformation")
    reference_function = _find_child(process_info, "referenceFunction")
    flow_data = _find_child(dataset, "flowData")
    return reference_function is not None and flow_data is not None


def _collect_ecospold_content(
    source: str,
    *,
    xml_error_policy: str = "fail",
) -> tuple[
    list[str],
    dict[str, DatasetLocator],
    list[DatasetEntry],
    dict[str, FlowInfo],
    dict[str, UnitInfo],
    dict[str, ImpactCategory],
    dict[str, DatasetEntry],
    dict[str, dict[str, int | str]],
    list[dict[str, str]],
]:
    if xml_error_policy not in ECOSPOLD_XML_ERROR_POLICIES:
        raise ValueError(f"Unsupported EcoSpold1 XML error policy: {xml_error_policy}")
    entry_paths: list[str] = []
    processes: dict[str, DatasetLocator] = {}
    entries: list[DatasetEntry] = []
    flows: dict[str, FlowInfo] = {}
    unit_names: set[str] = set()
    impact_categories: dict[str, ImpactCategory] = {}
    impact_methods: dict[str, DatasetEntry] = {}
    process_source_map: dict[str, dict[str, int | str]] = {}
    parse_warnings: list[dict[str, str]] = []

    found_ecospold = False
    for rel_path, raw_bytes in _iter_source_entries(source):
        entry_paths.append(rel_path)
        lower_rel_path = rel_path.lower()
        if not lower_rel_path.endswith((".xml", ".spold")):
            continue
        try:
            root = _try_parse_ecospold_root(raw_bytes, rel_path)
        except DataFormatError as exc:
            parse_warnings.append(
                {
                    "source_file": rel_path,
                    "message": str(exc),
                    "error_type": type(exc).__name__,
                }
            )
            if xml_error_policy == "warn_skip":
                continue
            raise
        if root is None:
            continue
        found_ecospold = True
        root_namespace = _namespace_uri(root.tag)
        for dataset_index, dataset in enumerate(_iter_children(root, "dataset")):
            key = _dataset_key(rel_path, dataset_index)
            if _dataset_is_impact_category(dataset, root_namespace=root_namespace):
                method_id, category, method_entry = _impact_category_from_dataset(
                    dataset,
                    rel_path=rel_path,
                    dataset_index=dataset_index,
                    unit_names=unit_names,
                )
                impact_categories[category.category_id] = category
                impact_methods.setdefault(method_id, method_entry)
                entries.append(
                    DatasetEntry(
                        path=key,
                        raw_bytes=b"",
                        data=category.raw,
                        object_type="impact_category",
                        object_id=category.category_id,
                        name=category.name,
                    )
                )
                continue
            if not _dataset_is_process_dataset(dataset, root_namespace=root_namespace):
                parse_warnings.append(
                    {
                        "source_file": rel_path,
                        "message": (
                            f"Skipped non-process EcoSpold1 dataset {key}: "
                            "the dataset is neither a process/activity inventory nor an impact-method dataset."
                        ),
                        "error_type": "EcoSpoldDatasetClassificationWarning",
                    }
                )
                continue

            locator, process_entry = _parse_process_dataset(
                dataset,
                rel_path=rel_path,
                dataset_index=dataset_index,
                flows=flows,
                unit_names=unit_names,
            )
            processes[locator.object_id] = locator
            process_source_map[locator.path] = {
                "entry_path": rel_path,
                "dataset_index": dataset_index,
            }
            entries.append(process_entry)

    if not found_ecospold:
        if parse_warnings:
            if xml_error_policy == "warn_skip":
                units = _parse_unit_registry(unit_names)
                return (
                    entry_paths,
                    processes,
                    entries,
                    flows,
                    units,
                    impact_categories,
                    impact_methods,
                    process_source_map,
                    parse_warnings,
                )
            raise DataFormatError(
                f"No valid EcoSpold1 datasets were found in {source}. "
                f"{len(parse_warnings)} XML file(s) failed to parse. "
                f"First error: {parse_warnings[0]['message']}"
            )
        raise DataFormatError(f"No EcoSpold1 datasets were found in {source}")

    units = _parse_unit_registry(unit_names)
    return (
        entry_paths,
        processes,
        entries,
        flows,
        units,
        impact_categories,
        impact_methods,
        process_source_map,
        parse_warnings,
    )


def _build_category_path_diagnostics(flows: Dict[str, FlowInfo]) -> CategoryPathDiagnostics:
    diagnostics = CategoryPathDiagnostics()
    elementary_flows = [flow for flow in flows.values() if flow.is_elementary]
    diagnostics.n_elementary_flows = len(elementary_flows)
    diagnostics.n_elementary_flows_with_category_path = sum(1 for flow in elementary_flows if flow.category_path.strip())
    diagnostics.n_unresolved_elementary_flows = (
        diagnostics.n_elementary_flows - diagnostics.n_elementary_flows_with_category_path
    )
    diagnostics.pct_elementary_flows_with_category_path = (
        100.0 * diagnostics.n_elementary_flows_with_category_path / diagnostics.n_elementary_flows
        if diagnostics.n_elementary_flows
        else 0.0
    )
    diagnostics.sample_unresolved_elementary_flows = [
        {
            "flow_id": flow.flow_id,
            "flow_name": flow.name,
            "source_file": "",
            "category": "",
            "category_reference_aliases": [],
            "raw_keys": sorted(str(key) for key in flow.raw.keys()),
        }
        for flow in elementary_flows
        if not flow.category_path.strip()
    ][:5]
    return diagnostics


def index_ecospold_archive(
    source: str,
    *,
    require_processes: bool = True,
    require_flows: bool = True,
    xml_error_policy: str = "fail",
) -> JsonLdArchiveIndex:
    (
        entry_paths,
        processes,
        _entries,
        flows,
        units,
        impact_categories,
        _impact_methods,
        process_source_map,
        parse_warnings,
    ) = _collect_ecospold_content(source, xml_error_policy=xml_error_policy)
    if require_processes and not processes:
        raise DataFormatError(f"No EcoSpold1 process datasets found in {source}")
    if require_flows and not flows:
        raise DataFormatError(f"No EcoSpold1 flows found in {source}")
    return JsonLdArchiveIndex(
        source_path=source,
        source_name=Path(source).name,
        resolved_source_path=source,
        is_zip=Path(source).suffix.lower() == ".zip",
        entry_paths=entry_paths,
        processes=processes,
        flows=flows,
        units=units,
        impact_categories=impact_categories,
        category_path_diagnostics=_build_category_path_diagnostics(flows),
        source_format="ecospold1",
        extra={
            "process_source_map": process_source_map,
            "parse_warnings": parse_warnings,
            "xml_error_policy": xml_error_policy,
        },
    )


def load_ecospold_archive(
    source: str,
    *,
    require_processes: bool = True,
    require_flows: bool = True,
    xml_error_policy: str = "fail",
) -> JsonLdArchive:
    (
        _entry_paths,
        processes,
        entries,
        flows,
        units,
        impact_categories,
        impact_methods,
        process_source_map,
        parse_warnings,
    ) = _collect_ecospold_content(source, xml_error_policy=xml_error_policy)
    if require_processes and not processes:
        raise DataFormatError(f"No EcoSpold1 process datasets found in {source}")
    if require_flows and not flows:
        raise DataFormatError(f"No EcoSpold1 flows found in {source}")
    process_entries = {entry.object_id: entry for entry in entries if entry.object_type == "process"}
    return JsonLdArchive(
        source_path=source,
        source_name=Path(source).name,
        is_zip=Path(source).suffix.lower() == ".zip",
        entries=entries,
        processes=process_entries,
        flows=flows,
        units=units,
        impact_categories=impact_categories,
        impact_methods=impact_methods,
        other_entries=[entry for entry in entries if entry.object_type != "process"],
        category_path_diagnostics=_build_category_path_diagnostics(flows),
        source_format="ecospold1",
        extra={
            "process_source_map": process_source_map,
            "parse_warnings": parse_warnings,
            "xml_error_policy": xml_error_policy,
        },
    )


def iter_ecospold_processes(index: JsonLdArchiveIndex) -> Iterator[tuple[DatasetLocator, dict[str, Any]]]:
    if index.source_format != "ecospold1":
        raise ValueError("iter_ecospold_processes requires an EcoSpold1 archive index")
    process_locator_by_key = {locator.path: locator for locator in index.processes.values()}
    relevant_entries = {
        str(meta["entry_path"])
        for meta in index.extra.get("process_source_map", {}).values()
        if isinstance(meta, dict) and meta.get("entry_path")
    }
    for rel_path, raw_bytes in _iter_source_entries(index.resolved_source_path):
        if relevant_entries and rel_path not in relevant_entries:
            continue
        lower_rel_path = rel_path.lower()
        if not lower_rel_path.endswith((".xml", ".spold")):
            continue
        root = _try_parse_ecospold_root(raw_bytes, rel_path)
        if root is None:
            continue
        for dataset_index, dataset in enumerate(_iter_children(root, "dataset")):
            key = _dataset_key(rel_path, dataset_index)
            locator = process_locator_by_key.get(key)
            if locator is None:
                continue
            _scratch_flows: dict[str, FlowInfo] = {}
            unit_names: set[str] = set()
            _locator, entry = _parse_process_dataset(
                dataset,
                rel_path=rel_path,
                dataset_index=dataset_index,
                flows=_scratch_flows,
                unit_names=unit_names,
            )
            yield locator, entry.data


def _serialise_xml(root: ET.Element) -> bytes:
    if root.tag.startswith("{"):
        ET.register_namespace("", root.tag.split("}", 1)[0][1:])
    ET.register_namespace("xsi", XSI_NAMESPACE)
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def write_reduced_ecospold_archive(
    source: str,
    destination_zip: Path,
    reduced_results: Dict[str, ProcessReductionResult],
    *,
    process_source_map: Dict[str, dict[str, int | str]],
) -> None:
    reduced_by_entry: dict[str, dict[int, ProcessReductionResult]] = defaultdict(dict)
    for key, result in reduced_results.items():
        source_meta = process_source_map.get(key)
        if not source_meta:
            continue
        entry_path = str(source_meta["entry_path"])
        dataset_index = int(source_meta["dataset_index"])
        reduced_by_entry[entry_path][dataset_index] = result

    with ZipFile(destination_zip, "w", compression=ZIP_DEFLATED) as output_archive:
        for rel_path, raw_bytes in _iter_source_entries(source):
            dataset_results = reduced_by_entry.get(rel_path)
            if not dataset_results or not rel_path.lower().endswith((".xml", ".spold")):
                output_archive.writestr(rel_path, raw_bytes)
                continue
            root = _try_parse_ecospold_root(raw_bytes, rel_path)
            if root is None:
                output_archive.writestr(rel_path, raw_bytes)
                continue
            for dataset_index, dataset in enumerate(_iter_children(root, "dataset")):
                result = dataset_results.get(dataset_index)
                if result is None:
                    continue
                flow_data = _find_child(dataset, "flowData")
                if flow_data is None:
                    continue
                exchanges = [child for child in _iter_children(flow_data, "exchange")]
                removed = set(result.removed_indices)
                for exchange_index, exchange_element in enumerate(exchanges):
                    if exchange_index in removed:
                        flow_data.remove(exchange_element)
            output_archive.writestr(rel_path, _serialise_xml(root))
