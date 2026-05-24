"""Metadata extraction for reduction curve comparison."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Callable, Iterable, Optional

from .contribution import exchange_flow_id
from .errors import RunCancelledError
from .jsonld_reader import index_archive, iter_source_entries, parse_json_object
from .models import TauReductionRun


CurveProgressCallback = Callable[[str, int, int], None]
CurveCancelCallback = Callable[[], bool]

_KNOWN_HASH_KEYS = (
    "input_database_hash",
    "input_database_sha256",
    "input_database_md5",
    "database_hash",
    "database_sha256",
)
_TAU_RE = re.compile(r"(?:^|[^0-9])tau[_-]?([0-9]*\.?[0-9]+)", re.IGNORECASE)


def clone_run(run: TauReductionRun) -> TauReductionRun:
    copied = replace(run)
    copied.warnings = list(run.warnings)
    copied.sourceWarnings = list(run.sourceWarnings)
    copied.groupWarnings = list(run.groupWarnings)
    return copied


def extract_run_metadata(
    source_path: str,
    *,
    progress_callback: Optional[CurveProgressCallback] = None,
    cancel_callback: Optional[CurveCancelCallback] = None,
) -> TauReductionRun:
    source = Path(source_path)
    run = TauReductionRun(
        id=source.name,
        sourceFileName=source.name,
        tau=None,
        elementaryBefore=None,
        elementaryAfter=None,
        elementaryRemoved=None,
        retainedPercent=None,
        removedPercent=None,
        validationStatus="unknown",
        sourcePath=str(source),
        runDirectory=str(_resolve_run_directory(source)),
        state="ready",
    )
    warnings: list[str] = []
    source_info = _load_source_summary(source)
    run.runDirectory = str(source_info["run_dir"])

    _emit(progress_callback, "Reading summary metadata", 1, 4)
    _check_cancel(cancel_callback)
    run_summary = _read_optional_json(source_info["summary_path"])

    if run_summary is not None:
        run.tau = _coerce_float(run_summary.get("tau"))
        run.inputDatabaseName = _input_database_name(run_summary)
        run.inputDatabaseHash = _input_database_hash(run_summary)
        run.validationStatus = _validation_status(run_summary)
        run.elementaryBefore = _coerce_int(run_summary.get("n_elementary_before"))
        run.elementaryAfter = _coerce_int(run_summary.get("n_elementary_after"))
        run.elementaryRemoved = _coerce_int(run_summary.get("n_elementary_removed"))

    _emit(progress_callback, "Reading manifests", 2, 4)
    _check_cancel(cancel_callback)
    manifest_counts = _manifest_counts(source_info["manifest_path"])
    if run.elementaryBefore is None:
        run.elementaryBefore = manifest_counts["before"]
    if run.elementaryAfter is None:
        run.elementaryAfter = manifest_counts["after"]
    if run.elementaryRemoved is None:
        run.elementaryRemoved = manifest_counts["removed"]

    _fill_missing_counts(run)

    if run.tau is None:
        run.tau = _tau_from_name(source.name)
        if run.tau is None and source_info["output_zip"] is not None:
            run.tau = _tau_from_name(source_info["output_zip"].name)
    if run.tau is None:
        warnings.append("Tau is missing.")

    if run.inputDatabaseName is None and source_info["output_zip"] is not None:
        run.inputDatabaseName = source_info["output_zip"].stem

    if run.elementaryAfter is None and source_info["output_zip"] is not None:
        _emit(progress_callback, "Scanning reduced database ZIP", 3, 4)
        run.elementaryAfter = _count_elementary_exchanges(
            source_info["output_zip"],
            progress_callback=progress_callback,
            cancel_callback=cancel_callback,
        )
        _fill_missing_counts(run)
    else:
        _emit(progress_callback, "Summary metadata loaded", 3, 4)

    _recompute_percentages(run)

    if run.elementaryBefore is None:
        warnings.append("Original elementary-exchange count is missing.")
    if run.elementaryAfter is None:
        warnings.append("Reduced elementary-exchange count is missing.")
    if run.validationStatus == "fail":
        warnings.append("Run summary indicates a failed reduction run.")
    if run_summary is None:
        warnings.append("`run_summary.json` was not found.")

    run.sourceWarnings = warnings
    run.refresh_warnings()
    _emit(progress_callback, "Metadata ready", 4, 4)
    return run


def group_warnings(runs: Iterable[TauReductionRun]) -> dict[str, list[str]]:
    warnings: dict[str, list[str]] = {}
    run_list = list(runs)
    before_counts = {run.elementaryBefore for run in run_list if run.elementaryBefore is not None}
    tau_counts: dict[float, int] = {}
    for run in run_list:
        if run.tau is not None:
            tau_counts[run.tau] = tau_counts.get(run.tau, 0) + 1

    inconsistent_before = len(before_counts) > 1
    for run in run_list:
        issues: list[str] = []
        if run.state in {"queued", "processing"}:
            warnings[run.id] = issues
            continue
        if inconsistent_before:
            issues.append("Inconsistent original elementary-exchange counts within this database group.")
        if run.tau is None:
            issues.append("Tau is missing, so this run is excluded from the curves.")
        elif tau_counts.get(run.tau, 0) > 1:
            issues.append("Tau is duplicated within this database group.")
        if run.validationStatus == "fail":
            issues.append("Coverage validation failed for this run.")
        if not curve_point_is_valid(run):
            issues.append("This run does not have enough metadata for curve plotting.")
        warnings[run.id] = issues
    return warnings


def curve_point_is_valid(run: TauReductionRun) -> bool:
    return (
        run.tau is not None
        and run.retainedPercent is not None
        and run.removedPercent is not None
        and run.validationStatus != "fail"
    )


def export_curve_rows(groups: Iterable[tuple[str, Iterable[TauReductionRun]]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for group_name, runs in groups:
        for run in runs:
            rows.append(
                {
                    "database_group": group_name,
                    "source_file_name": run.sourceFileName,
                    "tau": run.tau,
                    "elementary_before": run.elementaryBefore,
                    "elementary_after": run.elementaryAfter,
                    "elementary_removed": run.elementaryRemoved,
                    "retained_percent": run.retainedPercent,
                    "removed_percent": run.removedPercent,
                    "validation_status": run.validationStatus,
                    "input_database_name": run.inputDatabaseName or "",
                    "input_database_hash": run.inputDatabaseHash or "",
                    "warnings": " | ".join(run.warnings),
                }
            )
    return rows


def _resolve_run_directory(source: Path) -> Path:
    return source if source.is_dir() else source.parent


def _load_source_summary(source: Path) -> dict[str, Optional[Path]]:
    run_dir = _resolve_run_directory(source)
    output_zip: Path | None = None
    if source.is_file() and source.suffix.lower() in {".zip", ".zolca"}:
        output_zip = source
    summary_path = source if source.is_file() and source.name == "run_summary.json" else run_dir / "run_summary.json"
    manifest_path = (
        source if source.is_file() and source.name == "process_manifest.csv" else run_dir / "process_manifest.csv"
    )
    if summary_path.exists():
        summary = _read_optional_json(summary_path)
        output_value = summary.get("output_zip") if isinstance(summary, dict) else None
        if isinstance(output_value, str) and output_value.strip():
            candidate = Path(output_value)
            if not candidate.is_absolute():
                candidate = run_dir / candidate
            if candidate.exists():
                output_zip = candidate
    if output_zip is None:
        zips = sorted(run_dir.glob("*_lite_tau_*.zip"))
        if zips:
            output_zip = zips[0]
    return {
        "run_dir": run_dir,
        "summary_path": summary_path,
        "manifest_path": manifest_path,
        "output_zip": output_zip,
    }


def _read_optional_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _input_database_name(payload: dict) -> str | None:
    value = payload.get("input_database_name") or payload.get("input_database_zip")
    if not isinstance(value, str) or not value.strip():
        return None
    return Path(value).name


def _input_database_hash(payload: dict) -> str | None:
    for key in _KNOWN_HASH_KEYS:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _validation_status(payload: dict) -> str:
    if payload.get("status") == "failed" or payload.get("error"):
        return "fail"
    coverage_failures = _coerce_int(payload.get("n_coverage_failures"))
    if coverage_failures is not None and coverage_failures > 0:
        return "fail"
    if payload:
        return "pass"
    return "unknown"


def _manifest_counts(path: Path) -> dict[str, int | None]:
    if not path.exists():
        return {"before": None, "after": None, "removed": None}
    before = 0
    after = 0
    removed = 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            before += _coerce_int(row.get("n_elementary_before")) or 0
            after += _coerce_int(row.get("n_elementary_after")) or 0
            removed += _coerce_int(row.get("n_elementary_removed")) or 0
    return {
        "before": before,
        "after": after,
        "removed": removed,
    }


def _count_elementary_exchanges(
    archive_path: Path,
    *,
    progress_callback: Optional[CurveProgressCallback] = None,
    cancel_callback: Optional[CurveCancelCallback] = None,
) -> int:
    index = index_archive(str(archive_path), require_processes=True, require_flows=True, conversion_scope="inspect")
    process_paths = {locator.path for locator in index.processes.values()}
    process_total = max(len(process_paths), 1)
    elementary_after = 0
    processed = 0
    for rel_path, raw_bytes in iter_source_entries(index.resolved_source_path):
        _check_cancel(cancel_callback)
        if rel_path not in process_paths:
            continue
        processed += 1
        process_data = parse_json_object(raw_bytes, rel_path)
        for exchange in process_data.get("exchanges") or []:
            flow_id = exchange_flow_id(exchange)
            if not flow_id:
                continue
            flow = index.flows.get(flow_id)
            if flow is not None and flow.is_elementary:
                elementary_after += 1
        _emit(progress_callback, "Scanning reduced database ZIP", processed, process_total)
    return elementary_after


def _fill_missing_counts(run: TauReductionRun) -> None:
    if run.elementaryBefore is None and run.elementaryAfter is not None and run.elementaryRemoved is not None:
        run.elementaryBefore = run.elementaryAfter + run.elementaryRemoved
    if run.elementaryAfter is None and run.elementaryBefore is not None and run.elementaryRemoved is not None:
        run.elementaryAfter = run.elementaryBefore - run.elementaryRemoved
    if run.elementaryRemoved is None and run.elementaryBefore is not None and run.elementaryAfter is not None:
        run.elementaryRemoved = run.elementaryBefore - run.elementaryAfter


def _recompute_percentages(run: TauReductionRun) -> None:
    if run.elementaryBefore is None or run.elementaryAfter is None:
        return
    if run.elementaryBefore <= 0:
        return
    retained = 100.0 * run.elementaryAfter / run.elementaryBefore
    removed = 100.0 * (1.0 - (run.elementaryAfter / run.elementaryBefore))
    run.retainedPercent = round(retained, 12)
    run.removedPercent = round(removed, 12)


def _coerce_int(value: object) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _coerce_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _tau_from_name(name: str) -> float | None:
    match = _TAU_RE.search(name)
    if match is None:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _emit(progress_callback: Optional[CurveProgressCallback], message: str, current: int, total: int) -> None:
    if progress_callback is not None:
        progress_callback(message, current, total)


def _check_cancel(cancel_callback: Optional[CurveCancelCallback]) -> None:
    if cancel_callback is not None and cancel_callback():
        raise RunCancelledError("Curve metadata extraction cancelled.")
