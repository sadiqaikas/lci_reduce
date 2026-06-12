"""Tiny file-writing helpers for JSON and CSV sidecars.

These helpers are intentionally minimal, but keeping them central matters for
auditability: all workflows then write JSON and CSV sidecars in one consistent,
deterministic style.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .models import WarningRecord


def write_json(path: str | Path, payload: Any) -> str:
    """Write deterministic, pretty JSON and return the final path string.

    ``sort_keys=True`` is used so repeated runs produce stable file ordering,
    which helps both human diffing and automated regression tests.
    """
    target = Path(path)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return str(target)


def write_csv(path: str | Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> str:
    """Write a CSV table with a fixed header order and return the path.

    The writer fills missing fields with empty strings rather than omitting
    columns so the output schema remains rectangular and easy to audit in
    spreadsheets.
    """
    target = Path(path)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    return str(target)


def warnings_to_rows(warnings: list[WarningRecord]) -> list[dict[str, Any]]:
    """Serialise warning records into plain dictionaries for sidecar output.

    Warnings are carried as typed objects inside Python, then flattened only at
    the final reporting boundary.
    """
    return [warning.to_dict() for warning in warnings]
