#!/usr/bin/env python3
"""Fetch official GLAD flow-list CSVs and write reduced local snapshots."""

from __future__ import annotations

import base64
import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


REPO_API = "https://api.github.com/repos/UNEP-Economy-Division/GLAD-ElementaryFlowResources"
TARGETS = {
    "ecoinventEFv3.7": "Mapping/Input/Flowlists/ecoinventEFv3.7.csv",
    "ILCD_EFv3.0": "Mapping/Input/Flowlists/ILCD_EFv3.0.csv",
    "FEDEFLv1.0.3": "Mapping/Input/Flowlists/FEDEFLv1.0.3.csv",
    "IDEA_EFv2.3": "Mapping/Input/Flowlists/IDEA_EFv2.3.csv",
}
OUTPUT_COLUMNS = [
    "source_file",
    "list_name",
    "flow_uuid",
    "flow_name",
    "context",
    "unit",
    "cas_number",
    "synonyms",
]


def _fetch_json(url: str) -> dict:
    completed = subprocess.run(
        ["curl", "-L", url],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def _normalise_text(value: str | None) -> str:
    return " ".join(str(value or "").strip().split())


def _first(row: dict[str, str], *names: str) -> str:
    for name in names:
        if name in row and str(row[name]).strip():
            return str(row[name]).strip()
    return ""


def _decode_blob_via_contents(path: str) -> str:
    payload = _fetch_json(f"{REPO_API}/contents/{path}?ref=master")
    if payload.get("encoding") == "base64" and payload.get("content"):
        return base64.b64decode(payload["content"]).decode("utf-8-sig")
    blob_url = payload.get("git_url")
    if not blob_url:
        raise RuntimeError(f"No inline content or git blob URL for {path}")
    blob_payload = _fetch_json(blob_url)
    return base64.b64decode(blob_payload["content"]).decode("utf-8-sig")


def _snapshot_rows(list_name: str, source_file: str, csv_text: str) -> list[dict[str, str]]:
    reader = csv.DictReader(csv_text.splitlines())
    rows: list[dict[str, str]] = []
    for raw_row in reader:
        row = {str(key): str(value or "") for key, value in (raw_row or {}).items()}
        rows.append(
            {
                "source_file": source_file,
                "list_name": list_name,
                "flow_uuid": _normalise_text(_first(row, "Flow UUID")),
                "flow_name": _normalise_text(_first(row, "Flowable", "\ufeffFlowable")),
                "context": _normalise_text(_first(row, "Context")),
                "unit": _normalise_text(_first(row, "Unit")),
                "cas_number": _normalise_text(_first(row, "CAS No")),
                "synonyms": _normalise_text(_first(row, "Synonyms")),
            }
        )
    return rows


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    output_dir = root / "priority_glad"
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_targets: dict[str, dict[str, str | int]] = {}
    for list_name, upstream_path in TARGETS.items():
        csv_text = _decode_blob_via_contents(upstream_path)
        rows = _snapshot_rows(list_name, upstream_path, csv_text)
        output_path = output_dir / f"{list_name}.csv"
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        manifest_targets[list_name] = {
            "upstream_repo": "https://github.com/UNEP-Economy-Division/GLAD-ElementaryFlowResources",
            "upstream_file": upstream_path,
            "snapshot_file": output_path.name,
            "row_count": len(rows),
        }

    manifest = {
        "upstream_repo": "https://github.com/UNEP-Economy-Division/GLAD-ElementaryFlowResources",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "notes": [
            "GLAD flow-list CSV snapshots are the authoritative assets used for direct target matching.",
            "GLAD pairwise Excel mapping workbooks exist upstream but are not used as the primary matcher for arbitrary source databases.",
        ],
        "targets": manifest_targets,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
