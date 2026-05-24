"""Schema and object detection helpers."""

from __future__ import annotations

from typing import Any, Dict, Optional


PROCESS_TYPES = {"process"}
FLOW_TYPES = {"flow"}
UNIT_TYPES = {"unit"}
UNIT_GROUP_TYPES = {"unitgroup", "unit_group"}
IMPACT_CATEGORY_TYPES = {"impactcategory", "lciacategory"}
IMPACT_METHOD_TYPES = {"impactmethod", "lciamethod"}
CATEGORY_TYPES = {"category"}


def _normalise_token(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum() or ch == "_")


def detect_object_type(data: Dict[str, Any], path: str) -> str:
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
    lower_path = path.lower()
    if "/process" in lower_path or lower_path.startswith("process"):
        return "process"
    if "/flow" in lower_path or lower_path.startswith("flow"):
        return "flow"
    if "/categories/" in lower_path or lower_path.startswith("categories/"):
        return "category"
    if "/unit_group" in lower_path or "/unitgroups" in lower_path:
        return "unit_group"
    if "/unit" in lower_path or lower_path.startswith("unit"):
        return "unit"
    if "/lcia_method" in lower_path or "/impact_method" in lower_path:
        return "impact_method"
    if "/lcia_categor" in lower_path or "/impact_categor" in lower_path:
        return "impact_category"
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
    if isinstance(category, dict):
        if isinstance(category.get("path"), str):
            return category["path"]
        if isinstance(category.get("name"), str):
            return category["name"]
    if isinstance(data.get("categoryPath"), str):
        return data["categoryPath"]
    categories = data.get("categories")
    if isinstance(categories, list):
        tokens = []
        for item in categories:
            if isinstance(item, str):
                tokens.append(item)
            elif isinstance(item, dict) and isinstance(item.get("name"), str):
                tokens.append(item["name"])
        return "/".join(tokens)
    return ""
