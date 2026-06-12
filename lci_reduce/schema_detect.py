"""Schema and object detection helpers."""

from __future__ import annotations

from typing import Any, Dict, Optional


PROCESS_TYPES = {"process"}
FLOW_TYPES = {"flow"}
FLOW_PROPERTY_TYPES = {"flowproperty", "flow_property"}
UNIT_TYPES = {"unit"}
UNIT_GROUP_TYPES = {"unitgroup", "unit_group"}
IMPACT_CATEGORY_TYPES = {"impactcategory", "lciacategory"}
IMPACT_METHOD_TYPES = {"impactmethod", "lciamethod"}
CATEGORY_TYPES = {"category"}
TOP_LEVEL_OBJECT_TYPES = {
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
NON_DATASET_TOP_LEVEL_SEGMENTS = {"bin"}
RECOGNISED_COLLECTION_SEGMENTS = set(TOP_LEVEL_OBJECT_TYPES) | NON_DATASET_TOP_LEVEL_SEGMENTS


def _normalise_token(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum() or ch == "_")


def _path_segments(path: str) -> list[str]:
    parts = [part.strip() for part in path.replace("\\", "/").split("/") if part.strip()]
    return [part.casefold() for part in parts]


def _collection_segment(path: str) -> str:
    parts = _path_segments(path)
    if not parts:
        return ""
    if parts[0] in RECOGNISED_COLLECTION_SEGMENTS:
        return parts[0]
    if len(parts) >= 2 and parts[1] in RECOGNISED_COLLECTION_SEGMENTS:
        return parts[1]
    return parts[0]


def _explicit_object_type(data: Dict[str, Any]) -> str:
    candidates = [
        data.get("@type"),
        data.get("type"),
        data.get("olcaType"),
    ]
    for candidate in candidates:
        if isinstance(candidate, str):
            token = _normalise_token(candidate)
            if token in PROCESS_TYPES:
                return "process"
            if token in FLOW_TYPES:
                return "flow"
            if token in FLOW_PROPERTY_TYPES:
                return "flow_property"
            if token in UNIT_TYPES:
                return "unit"
            if token in UNIT_GROUP_TYPES:
                return "unit_group"
            if token in IMPACT_CATEGORY_TYPES:
                return "impact_category"
            if token in IMPACT_METHOD_TYPES:
                return "impact_method"
            if token in CATEGORY_TYPES:
                return "category"
    return ""


def detect_object_type(data: Dict[str, Any], path: str) -> str:
    explicit_type = _explicit_object_type(data)
    collection = _collection_segment(path)
    path_type = TOP_LEVEL_OBJECT_TYPES.get(collection, "")
    if explicit_type:
        if collection in NON_DATASET_TOP_LEVEL_SEGMENTS:
            return "other"
        if path_type and path_type != explicit_type:
            return "other"
        return explicit_type
    if path_type:
        return path_type
    return "other"


def extract_object_id(data: Dict[str, Any]) -> Optional[str]:
    for key in ("@id", "id", "uuid", "refId"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def extract_name(data: Any) -> str:
    if not isinstance(data, dict):
        return ""
    for key in ("name", "label", "description"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def reference_id(ref: Any) -> Optional[str]:
    if ref is None:
        return None
    if isinstance(ref, str):
        return ref
    if isinstance(ref, dict):
        for key in ("@id", "id", "refId", "uuid"):
            value = ref.get(key)
            if isinstance(value, str) and value.strip():
                return value
    return None


def reference_name(ref: Any) -> str:
    if isinstance(ref, dict):
        value = ref.get("name")
        if isinstance(value, str):
            return value
    return ""




def category_path_text(data: Dict[str, Any]) -> str:
    category = data.get("category")

    if isinstance(category, str) and category.strip():
        return category.strip()

    if isinstance(category, dict):
        if isinstance(category.get("path"), str) and category["path"].strip():
            return category["path"].strip()
        if isinstance(category.get("categoryPath"), str) and category["categoryPath"].strip():
            return category["categoryPath"].strip()
        if isinstance(category.get("name"), str) and category["name"].strip():
            return category["name"].strip()

    if isinstance(data.get("categoryPath"), str) and data["categoryPath"].strip():
        return data["categoryPath"].strip()

    categories = data.get("categories")
    if isinstance(categories, list):
        tokens = []
        for item in categories:
            if isinstance(item, str) and item.strip():
                tokens.append(item.strip())
            elif isinstance(item, dict) and isinstance(item.get("name"), str) and item["name"].strip():
                tokens.append(item["name"].strip())
        return "/".join(tokens)

    return ""
