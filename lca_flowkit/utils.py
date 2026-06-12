"""Shared low-level helpers used across readers and workflows.

These helpers are intentionally small, but they are still scientifically
important because they define the common normalisation rules used everywhere
else in the package:

- how text is normalised before matching;
- how IDs are recovered from heterogeneous payloads;
- how inputs are classified as JSON-LD or EcoSpold1;
- how location payloads are turned into a common representation.

Keeping those rules in one module avoids subtle drift between readers and
workflows.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable
from zipfile import BadZipFile, ZipFile

from .errors import DataFormatError
from .models import Location, ProgressCallback, ProgressEvent


def emit_progress(progress: ProgressCallback | None, *, step: str, message: str, current: int, total: int) -> None:
    """Send a lightweight progress event when the caller provided a callback.

    Workflows accept an optional callback instead of depending on any GUI or CLI
    framework.  That keeps the package usable from scripts, notebooks, and
    tests while still allowing external callers to display progress.
    """
    if progress is None:
        return
    progress(ProgressEvent(step=step, message=message, current=current, total=total))


def normalise_text(value: str | None) -> str:
    """Collapse whitespace while preserving the original spelling.

    This is a gentle text normaliser.  It trims leading/trailing whitespace and
    compresses repeated internal whitespace, but it does not case-fold or alter
    spelling.
    """
    if value is None:
        return ""
    return " ".join(value.strip().split())


def fold_text(value: str | None) -> str:
    """Normalise and case-fold text for tolerant matching.

    ``casefold`` is stronger than ``lower`` for Unicode text.  Using it here
    makes name matching slightly more robust while still being deterministic.
    """
    return normalise_text(value).casefold()


def object_id(data: dict[str, Any], fallback: str = "") -> str:
    """Read the first usable identifier from common JSON-LD object fields.

    Different exports use different identifier field names.  The helper makes
    that variability explicit in one place instead of scattering it across the
    codebase.
    """
    for key in ("@id", "id", "uuid", "refId"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return fallback


def reference_id(value: Any) -> str:
    """Resolve an ID from either a raw string or a reference object.

    openLCA-style references are often stored either as ``{"@id": ...}`` style
    objects or as plain strings.  The helper accepts both shapes.
    """
    if isinstance(value, str):
        return value.strip()
    if not isinstance(value, dict):
        return ""
    return object_id(value)


def object_name(value: Any) -> str:
    """Resolve a display name from a raw string or a reference object.

    Returning a normalised display name is useful for human-readable warnings
    and metadata, but downstream matching still prefers stable IDs whenever they
    exist.
    """
    if isinstance(value, str):
        return normalise_text(value)
    if not isinstance(value, dict):
        return ""
    name = value.get("name")
    return normalise_text(name if isinstance(name, str) else "")


def parse_json_object(raw_bytes: bytes, rel_path: str) -> dict[str, Any]:
    """Parse one JSON-LD object and fail loudly on non-object payloads.

    The readers assume each JSON file contains one object-like record.  A
    top-level list or scalar would not fit the package's object model, so the
    parser rejects it clearly.
    """
    try:
        decoded = raw_bytes.decode("utf-8-sig")
        parsed = json.loads(decoded)
    except Exception as exc:  # pragma: no cover - defensive
        raise DataFormatError(f"Failed to parse JSON object {rel_path}: {exc}") from exc
    if not isinstance(parsed, dict):
        raise DataFormatError(f"JSON file {rel_path} must contain a top-level object")
    return parsed


def iter_source_entries(source: str) -> list[tuple[str, bytes]]:
    """Read every file from a folder, ZIP, or single-file input source.

    The package supports:

    - directory trees;
    - ZIP and ``.zolca`` archives;
    - single JSON/XML files for focused tests and small method inputs.

    The returned list always contains ``(relative_path, raw_bytes)`` pairs so
    the later readers can work identically regardless of storage container.
    """
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(source)
    if path.is_dir():
        return [
            (item.relative_to(path).as_posix(), item.read_bytes())
            for item in sorted(path.rglob("*"))
            if item.is_file()
        ]
    suffix = path.suffix.lower()
    if suffix in {".zip", ".zolca"}:
        try:
            with ZipFile(path, "r") as archive:
                return [
                    (name, archive.read(name))
                    for name in sorted(archive.namelist())
                    if not name.endswith("/")
                ]
        except BadZipFile as exc:
            raise DataFormatError(f"Failed to open archive {source}: {exc}") from exc
    if suffix in {".json", ".jsonld", ".xml", ".spold"}:
        return [(path.name, path.read_bytes())]
    raise DataFormatError(
        f"Unsupported input source: {source}. Supported inputs are folders, .zip archives, .zolca archives, .json files, .xml files, and .spold files."
    )


def detect_source_family(entries: Iterable[tuple[str, bytes]]) -> str:
    """Classify a source bundle as JSON-LD or EcoSpold1 before parsing.

    Mixed JSON and XML bundles are rejected because the rest of the package
    expects one coherent source family per call.
    """
    has_json = False
    has_xml = False
    for rel_path, _raw in entries:
        lowered = rel_path.lower()
        if lowered.endswith((".json", ".jsonld")):
            has_json = True
        if lowered.endswith((".xml", ".spold")):
            has_xml = True
    if has_json and not has_xml:
        return "jsonld"
    if has_xml and not has_json:
        return "ecospold1"
    if has_json and has_xml:
        raise DataFormatError("Mixed JSON and XML inputs are not supported in one source bundle")
    raise DataFormatError("No supported data files were found in the input source")


def split_category_path(path: str | None) -> tuple[str, str]:
    """Split a category path into compartment and subcompartment.

    The helper strips the common openLCA "Elementary flows" root label because
    it is a tree container, not the environmental compartment itself.

    Example:

    ``"Elementary flows/air/urban air close to ground"``

    becomes:

    ``("air", "urban air close to ground")``
    """
    parts = [part.strip() for part in normalise_text(path).split("/") if part.strip()]
    if parts and parts[0].casefold() in {"elementary flows", "elementary flow"}:
        parts = parts[1:]
    if not parts:
        return "", ""
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], "/".join(parts[1:])


def location_from_any(value: Any) -> Location:
    """Create a normalised ``Location`` object from a loose JSON payload.

    The package carries three parallel location hints:

    - a concrete ID, when one exists;
    - a human-readable name;
    - a short region/code token.

    Later scenario logic uses all three as candidate matching tokens.
    """
    if not isinstance(value, dict):
        return Location()
    return Location(
        location_id=reference_id(value),
        name=object_name(value),
        region=normalise_text(
            value.get("code") if isinstance(value.get("code"), str) else value.get("region") if isinstance(value.get("region"), str) else object_name(value)
        ),
    )


def unique_preserve_order(values: Iterable[str]) -> list[str]:
    """Return de-duplicated values without disturbing first-seen order.

    This helper is used heavily in scenario construction, where preserving the
    first-seen order makes output rows stable and easier to audit.
    """
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        token = str(value)
        if token in seen:
            continue
        seen.add(token)
        result.append(token)
    return result
