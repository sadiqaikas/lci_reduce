"""JSON-LD input reading and inspection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Optional
from zipfile import BadZipFile, ZipFile

from .errors import DataFormatError, NativeArchiveConversionError
from .lcia import (
    build_impact_category,
    build_method_category_lookup,
    collect_categories,
    impact_category_report,
    resolve_lcia_archives,
)
from .models import DatasetEntry, DatasetLocator, FlowInfo, InspectResult, JsonLdArchive, JsonLdArchiveIndex, UnitInfo
from .native_archive import get_cached_native_archive_jsonld
from .schema_detect import category_path_text, detect_object_type, extract_name, extract_object_id, reference_id


ARCHIVE_SUFFIXES = {".zip", ".zolca"}
NATIVE_OPENLCA_MARKERS = {
    "service.properties",
    "readme_do_not_touch_files.txt",
    "database.properties",
    "database.script",
    "database.data",
    "database.index",
    "database.log",
}


def _unsupported_source_message(source: str) -> str | None:
    suffix = Path(source).suffix.lower()
    if suffix == ".sip":
        return (
            f"Unsupported input source: {source}. SimaPro .sip packages are not supported by lci_reduce. "
            "Export the source data to openLCA JSON-LD ZIP first."
        )
    return None


def _looks_like_native_openlca_archive(entry_names: list[str]) -> bool:
    names = {Path(name).name.lower() for name in entry_names}
    if names & NATIVE_OPENLCA_MARKERS:
        return True
    paths = {name.lower() for name in entry_names}
    return any(path.startswith("seg0/") for path in paths) and any(path.startswith("log/") for path in paths)


def _no_jsonld_error(source: str, entry_names: list[str]) -> DataFormatError:
    suffix = Path(source).suffix.lower()
    if _looks_like_native_openlca_archive(entry_names):
        archive_label = ".zolca database backup" if suffix == ".zolca" else "database backup archive"
        return DataFormatError(
            f"No JSON-LD objects were found in {source}. "
            f"This looks like a native openLCA {archive_label} rather than a JSON-LD exchange archive. "
            "The archive layout matches the Derby-backed openLCA database format. "
            "lci_reduce only reads openLCA JSON-LD datasets packaged in a folder, .zip archive, or JSON-LD .zolca wrapper."
        )
    return DataFormatError(f"No JSON-LD objects were found in {source}")


def _iter_source_entries(source: str) -> Iterator[tuple[str, bytes]]:
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(source)
    unsupported_message = _unsupported_source_message(source)
    if unsupported_message is not None:
        raise DataFormatError(unsupported_message)
    if path.is_dir():
        for file_path in sorted(path.rglob("*")):
            if file_path.is_file():
                rel_path = file_path.relative_to(path).as_posix()
                yield rel_path, file_path.read_bytes()
        return
    if path.suffix.lower() in ARCHIVE_SUFFIXES:
        try:
            with ZipFile(path, "r") as archive:
                for name in sorted(archive.namelist()):
                    if name.endswith("/"):
                        continue
                    yield name, archive.read(name)
        except BadZipFile as exc:
            raise DataFormatError(f"Failed to open archive {source}: {exc}") from exc
        return
    raise DataFormatError(
        f"Unsupported input source: {source}. Supported inputs are JSON-LD folders, .zip archives, and .zolca archives."
    )


def parse_json_object(raw_bytes: bytes, rel_path: str) -> dict:
    try:
        data = json.loads(raw_bytes.decode("utf-8-sig"))
    except Exception as exc:  # pragma: no cover - defensive
        raise DataFormatError(f"Failed to parse JSON-LD file {rel_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise DataFormatError(f"JSON-LD file {rel_path} must contain a top-level JSON object")
    return data


def _load_json_entries(source: str) -> tuple[list[DatasetEntry], bool, list[str]]:
    entries: list[DatasetEntry] = []
    path = Path(source)
    entry_names: list[str] = []
    for rel_path, raw_bytes in _iter_source_entries(source):
        entry_names.append(rel_path)
        if not rel_path.lower().endswith((".json", ".jsonld")):
            continue
        data = parse_json_object(raw_bytes, rel_path)
        object_type = detect_object_type(data, rel_path)
        object_id = extract_object_id(data) or rel_path
        name = extract_name(data)
        entries.append(
            DatasetEntry(
                path=rel_path,
                raw_bytes=raw_bytes,
                data=data,
                object_type=object_type,
                object_id=object_id,
                name=name,
            )
        )
    return entries, path.suffix.lower() in ARCHIVE_SUFFIXES, entry_names


def _explicit_category_path(data: dict) -> str:
    direct_path = data.get("categoryPath")
    if isinstance(direct_path, str) and direct_path.strip():
        return direct_path.strip()
    direct_path = data.get("path")
    if isinstance(direct_path, str) and direct_path.strip():
        return direct_path.strip()
    categories = data.get("categories")
    if isinstance(categories, list):
        tokens: list[str] = []
        for item in categories:
            if isinstance(item, str) and item.strip():
                tokens.append(item.strip())
            elif isinstance(item, dict):
                name = extract_name(item)
                if name:
                    tokens.append(name.strip())
        if tokens:
            return "/".join(tokens)
    category = data.get("category")
    if isinstance(category, dict):
        parent_path = category.get("path")
        if isinstance(parent_path, str) and parent_path.strip():
            return parent_path.strip()
    return ""


def _join_category_path(parent_path: str, name: str) -> str:
    parent = parent_path.strip().strip("/")
    child = name.strip().strip("/")
    if parent and child:
        return f"{parent}/{child}"
    return parent or child


def _category_parent_id(data: dict) -> Optional[str]:
    for key in ("parentCategory", "category", "parent"):
        parent_id = reference_id(data.get(key))
        if parent_id:
            return parent_id
    return None


def _build_category_path_lookup(entries: list[DatasetEntry]) -> dict[str, str]:
    category_entries = {entry.object_id: entry for entry in entries if entry.object_type == "category"}
    if not category_entries:
        return {}
    cache: dict[str, str] = {}
    visiting: set[str] = set()

    def resolve(category_id: str) -> str:
        if category_id in cache:
            return cache[category_id]
        if category_id in visiting:
            return ""
        entry = category_entries.get(category_id)
        if entry is None:
            return ""
        visiting.add(category_id)
        explicit_path = _explicit_category_path(entry.data)
        name = (entry.name or extract_name(entry.data) or "").strip()
        if explicit_path:
            path = explicit_path
            if name and explicit_path != name and not explicit_path.endswith(f"/{name}"):
                path = _join_category_path(explicit_path, name)
        else:
            parent_path = resolve(_category_parent_id(entry.data) or "")
            path = _join_category_path(parent_path, name)
        visiting.remove(category_id)
        cache[category_id] = path
        return path

    for category_id in category_entries:
        resolve(category_id)
    return cache


def _flow_info(entry: DatasetEntry, category_path_lookup: Optional[dict[str, str]] = None) -> FlowInfo:
    flow_type = str(entry.data.get("flowType") or entry.data.get("flow_type") or "")
    is_elementary = flow_type.upper() == "ELEMENTARY_FLOW"
    category = entry.data.get("category")
    category_path = _explicit_category_path(entry.data)
    category_name = extract_name(category) if isinstance(category, dict) else ""
    category_ref_id = reference_id(category)
    resolved_category_path = category_path_lookup.get(category_ref_id, "") if category_path_lookup and category_ref_id else ""
    if resolved_category_path and (
        not category_path
        or category_path == category_name
        or (category_name and category_path != resolved_category_path and not category_path.endswith(f"/{category_name}"))
    ):
        category_path = resolved_category_path
    if not category_path:
        category_path = category_path_text(entry.data)
    location = entry.data.get("location")
    location_id = None
    location_name = None
    location_region = None
    if isinstance(location, dict):
        location_id = location.get("@id") or location.get("id") or location.get("refId") or location.get("uuid")
        location_name = location.get("name")
        location_region = location.get("code") or location.get("region") or location_name
    reference_flow_property_id = None
    reference_flow_property_name = None
    direct_reference_property = entry.data.get("referenceFlowProperty")
    if isinstance(direct_reference_property, dict):
        reference_flow_property_id = reference_id(direct_reference_property)
        reference_flow_property_name = extract_name(direct_reference_property) or None
    if reference_flow_property_id is None:
        for flow_property_factor in entry.data.get("flowProperties", []) or []:
            if not isinstance(flow_property_factor, dict):
                continue
            if not (
                flow_property_factor.get("referenceFlowProperty")
                or flow_property_factor.get("isReferenceFlowProperty")
                or flow_property_factor.get("reference")
            ):
                continue
            flow_property = flow_property_factor.get("flowProperty")
            reference_flow_property_id = reference_id(flow_property)
            reference_flow_property_name = extract_name(flow_property) or None
            break
    return FlowInfo(
        flow_id=entry.object_id,
        name=entry.name,
        flow_type=flow_type,
        category_path=category_path,
        is_elementary=is_elementary,
        raw=entry.data,
        location_id=str(location_id) if location_id else None,
        location_name=str(location_name) if location_name else None,
        location_region=str(location_region) if location_region else None,
        reference_flow_property_id=str(reference_flow_property_id) if reference_flow_property_id else None,
        reference_flow_property_name=str(reference_flow_property_name) if reference_flow_property_name else None,
    )


def _extract_unit_group_flow_property(data: dict) -> tuple[Optional[str], Optional[str]]:
    for key in ("defaultFlowProperty", "referenceFlowProperty", "flowProperty"):
        flow_property = data.get(key)
        flow_property_id = reference_id(flow_property)
        flow_property_name = extract_name(flow_property)
        if flow_property_id or flow_property_name:
            return flow_property_id, flow_property_name or None
    return None, None


def _extract_units(entries: list[DatasetEntry]) -> dict[str, UnitInfo]:
    units: dict[str, UnitInfo] = {}
    for entry in entries:
        if entry.object_type == "unit":
            group_id = reference_id(entry.data.get("unitGroup"))
            flow_property_id, flow_property_name = _extract_unit_group_flow_property(entry.data)
            units[entry.object_id] = UnitInfo(
                unit_id=entry.object_id,
                name=entry.name,
                group_id=group_id,
                raw=entry.data,
                group_name=extract_name(entry.data.get("unitGroup")) or None,
                conversion_factor=float(entry.data["conversionFactor"]) if entry.data.get("conversionFactor") is not None else None,
                is_reference_unit=bool(entry.data.get("referenceUnit") or entry.data.get("isReferenceUnit")),
                flow_property_id=flow_property_id,
                flow_property_name=flow_property_name,
            )
        elif entry.object_type == "unit_group":
            group_flow_property_id, group_flow_property_name = _extract_unit_group_flow_property(entry.data)
            reference_unit_id = reference_id(entry.data.get("referenceUnit"))
            for unit_data in entry.data.get("units", []):
                unit_id = unit_data.get("@id") or unit_data.get("id") or unit_data.get("uuid")
                if not unit_id:
                    continue
                conversion_factor = unit_data.get("conversionFactor")
                if conversion_factor is None and reference_unit_id and str(unit_id) == str(reference_unit_id):
                    conversion_factor = 1.0
                units[unit_id] = UnitInfo(
                    unit_id=str(unit_id),
                    name=str(unit_data.get("name") or ""),
                    group_id=entry.object_id,
                    raw=unit_data,
                    group_name=entry.name or None,
                    conversion_factor=float(conversion_factor) if conversion_factor is not None else None,
                    is_reference_unit=bool(
                        unit_data.get("referenceUnit")
                        or unit_data.get("isReferenceUnit")
                        or (reference_unit_id and str(unit_id) == str(reference_unit_id))
                    ),
                    flow_property_id=group_flow_property_id,
                    flow_property_name=group_flow_property_name,
                )
    return units


def iter_source_entries(source: str) -> Iterator[tuple[str, bytes]]:
    return _iter_source_entries(source)


def read_source_bytes(source: str, rel_path: str) -> bytes:
    for entry_path, raw_bytes in _iter_source_entries(source):
        if entry_path == rel_path:
            return raw_bytes
    raise FileNotFoundError(f"{rel_path} not found in {source}")


def read_json_object(source: str, rel_path: str) -> dict:
    return parse_json_object(read_source_bytes(source, rel_path), rel_path)


def _build_archive_index(source: str) -> tuple[JsonLdArchiveIndex, list[str]]:
    processes: dict[str, DatasetLocator] = {}
    flows: dict[str, FlowInfo] = {}
    flow_entries: list[DatasetEntry] = []
    category_object_entries: list[DatasetEntry] = []
    units_entries: list[DatasetEntry] = []
    method_entries: dict[str, DatasetEntry] = {}
    category_entries: list[DatasetEntry] = []
    entry_paths: list[str] = []
    entry_names: list[str] = []
    path = Path(source)

    for rel_path, raw_bytes in _iter_source_entries(source):
        entry_names.append(rel_path)
        entry_paths.append(rel_path)
        if not rel_path.lower().endswith((".json", ".jsonld")):
            continue
        data = parse_json_object(raw_bytes, rel_path)
        object_type = detect_object_type(data, rel_path)
        object_id = extract_object_id(data) or rel_path
        name = extract_name(data)
        entry = DatasetEntry(
            path=rel_path,
            raw_bytes=raw_bytes,
            data=data,
            object_type=object_type,
            object_id=object_id,
            name=name,
        )
        if object_type == "process":
            processes[object_id] = DatasetLocator(
                path=rel_path,
                object_type=object_type,
                object_id=object_id,
                name=name,
            )
        elif object_type == "flow":
            flow_entries.append(entry)
        elif object_type == "category":
            category_object_entries.append(entry)
        elif object_type in {"unit", "unit_group"}:
            units_entries.append(entry)
        elif object_type == "impact_method":
            method_entries[object_id] = entry
        elif object_type == "impact_category":
            category_entries.append(entry)

    category_path_lookup = _build_category_path_lookup(category_object_entries)
    flows = {entry.object_id: _flow_info(entry, category_path_lookup) for entry in flow_entries}
    units = _extract_units(units_entries)
    method_lookup = {
        method_id: {
            **entry.data,
            "__source_file": entry.path,
        }
        for method_id, entry in method_entries.items()
    }
    method_category_lookup = build_method_category_lookup(method_lookup)
    categories = {
        entry.object_id: build_impact_category(entry.data, method_lookup, method_category_lookup, entry.path)
        for entry in category_entries
    }

    return (
        JsonLdArchiveIndex(
            source_path=source,
            source_name=path.name,
            resolved_source_path=source,
            is_zip=path.suffix.lower() in ARCHIVE_SUFFIXES,
            entry_paths=entry_paths,
            processes=processes,
            flows=flows,
            units=units,
            impact_categories=categories,
        ),
        entry_names,
    )


def index_archive(
    source: str,
    *,
    require_processes: bool = True,
    require_flows: bool = True,
    conversion_scope: str = "full",
) -> JsonLdArchiveIndex:
    index, entry_names = _build_archive_index(source)
    if not index.processes and not index.flows and not index.impact_categories:
        if _looks_like_native_openlca_archive(entry_names):
            try:
                cached_jsonld_zip = get_cached_native_archive_jsonld(source, scope=conversion_scope)
                index, _ = _build_archive_index(cached_jsonld_zip)
                index.source_path = source
                index.source_name = Path(source).name
                index.resolved_source_path = cached_jsonld_zip
            except NativeArchiveConversionError as exc:
                raise DataFormatError(str(exc)) from exc
        elif not any(name.lower().endswith((".json", ".jsonld")) for name in entry_names):
            raise _no_jsonld_error(source, entry_names)
    if require_processes and not index.processes:
        raise DataFormatError(f"No process objects found in {source}")
    if require_flows and not index.flows:
        raise DataFormatError(f"No flow objects found in {source}")
    return index


def load_archive(
    source: str,
    *,
    require_processes: bool = True,
    require_flows: bool = True,
    conversion_scope: str = "full",
) -> JsonLdArchive:
    entries, is_zip, entry_names = _load_json_entries(source)
    if not entries:
        if _looks_like_native_openlca_archive(entry_names):
            try:
                cached_jsonld_zip = get_cached_native_archive_jsonld(source, scope=conversion_scope)
                entries, _, _ = _load_json_entries(cached_jsonld_zip)
            except NativeArchiveConversionError as exc:
                raise DataFormatError(str(exc)) from exc
        if not entries:
            raise _no_jsonld_error(source, entry_names)
    processes = {entry.object_id: entry for entry in entries if entry.object_type == "process"}
    category_path_lookup = _build_category_path_lookup(entries)
    flows = {entry.object_id: _flow_info(entry, category_path_lookup) for entry in entries if entry.object_type == "flow"}
    units = _extract_units(entries)
    methods = {entry.object_id: entry for entry in entries if entry.object_type == "impact_method"}
    method_lookup = {
        method_id: {
            **entry.data,
            "__source_file": entry.path,
        }
        for method_id, entry in methods.items()
    }
    method_category_lookup = build_method_category_lookup(method_lookup)
    categories = {
        entry.object_id: build_impact_category(entry.data, method_lookup, method_category_lookup, entry.path)
        for entry in entries
        if entry.object_type == "impact_category"
    }
    if require_processes and not processes:
        raise DataFormatError(f"No process objects found in {source}")
    if require_flows and not flows:
        raise DataFormatError(f"No flow objects found in {source}")
    return JsonLdArchive(
        source_path=source,
        source_name=Path(source).name,
        is_zip=is_zip,
        entries=entries,
        processes=processes,
        flows=flows,
        units=units,
        impact_categories=categories,
        impact_methods=methods,
        other_entries=[entry for entry in entries if entry.object_type != "process"],
    )


def guess_method_families(method_names: list[str]) -> list[str]:
    families: set[str] = set()
    for name in method_names:
        stripped = name.strip()
        if not stripped:
            continue
        families.add(stripped.split()[0])
        if "-" in stripped:
            families.add(stripped.split("-", 1)[0].strip())
    return sorted(family for family in families if family)


def inspect_archives(database_archive: JsonLdArchive, methods_archive: JsonLdArchive | None = None) -> InspectResult:
    archives, _lcia_method_source, _internal_lcia_methods_ignored = resolve_lcia_archives(
        database_archive,
        methods_archive,
    )
    categories = collect_categories(archives)
    method_names = sorted(
        {
            category.method_name
            for category in categories
            if category.method_name
        }
    )
    categories_with_factors = sum(1 for category in categories if category.factor_candidates or category.factors)
    category_details = [impact_category_report(category) for category in categories]
    empty_categories = [impact_category_report(category) for category in categories if not (category.factor_candidates or category.factors)]
    return InspectResult(
        n_processes=len(database_archive.processes),
        n_flows=len(database_archive.flows),
        n_elementary_flows=sum(1 for flow in database_archive.flows.values() if flow.is_elementary),
        n_methods=len({category.method_id for category in categories if category.method_id}),
        n_categories=len(categories),
        database_methods=len({category.method_id for category in database_archive.impact_categories.values() if category.method_id}),
        database_categories=len(database_archive.impact_categories),
        database_contains_impact_methods=bool(database_archive.impact_categories),
        method_names=method_names,
        family_names=guess_method_families(method_names),
        cf_mappings_buildable=categories_with_factors > 0,
        categories_with_factors=categories_with_factors,
        lcia_categories=category_details,
        empty_lcia_categories=empty_categories,
    )
