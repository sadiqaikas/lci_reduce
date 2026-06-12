import csv
import json
from pathlib import Path

from lci_reduce.priority_glad import (
    build_priority_glad_rows,
    classify_unresolved_row,
    load_asset_manifest,
    match_priority_row_to_target,
    priority_glad_output_columns,
    write_priority_glad_outputs,
)


def _write_priority_csv(path: Path) -> Path:
    fieldnames = [
        "flow_id",
        "flow_name",
        "compartment",
        "subcompartment",
        "reference_unit",
        "occurrence_count",
        "characterised_occurrence_count",
        "tau_entry_min",
        "tau_entry_median",
        "tau_entry_max",
        "eta_0_95",
        "loss_max_0_95",
        "eta_0_99",
        "loss_max_0_99",
    ]
    rows = [
        {
            "flow_id": "uuid-consensus",
            "flow_name": "Carbon dioxide",
            "compartment": "air",
            "subcompartment": "urban air",
            "reference_unit": "kg",
            "occurrence_count": "5",
            "characterised_occurrence_count": "5",
            "tau_entry_min": "0.1",
            "tau_entry_median": "0.2",
            "tau_entry_max": "0.4",
            "eta_0_95": "0",
            "loss_max_0_95": "0",
            "eta_0_99": "0.1",
            "loss_max_0_99": "0.1",
        },
        {
            "flow_id": "uuid-only",
            "flow_name": "Different source name",
            "compartment": "air",
            "subcompartment": "urban air",
            "reference_unit": "kg",
            "occurrence_count": "4",
            "characterised_occurrence_count": "4",
            "tau_entry_min": "0.1",
            "tau_entry_median": "0.2",
            "tau_entry_max": "0.3",
            "eta_0_95": "0",
            "loss_max_0_95": "0.2",
            "eta_0_99": "0",
            "loss_max_0_99": "0.2",
        },
        {
            "flow_id": "source-no-uuid",
            "flow_name": "Methane",
            "compartment": "air",
            "subcompartment": "urban air",
            "reference_unit": "kg",
            "occurrence_count": "3",
            "characterised_occurrence_count": "3",
            "tau_entry_min": "0.6",
            "tau_entry_median": "0.7",
            "tau_entry_max": "0.8",
            "eta_0_95": "0.2",
            "loss_max_0_95": "0.4",
            "eta_0_99": "0.3",
            "loss_max_0_99": "0.4",
        },
        {
            "flow_id": "uuid-conflict",
            "flow_name": "Flow by name",
            "compartment": "soil",
            "subcompartment": "agricultural",
            "reference_unit": "kg",
            "occurrence_count": "2",
            "characterised_occurrence_count": "2",
            "tau_entry_min": "0.5",
            "tau_entry_median": "0.5",
            "tau_entry_max": "0.5",
            "eta_0_95": "0",
            "loss_max_0_95": "0.3",
            "eta_0_99": "0",
            "loss_max_0_99": "0.3",
        },
        {
            "flow_id": "source-ambiguous",
            "flow_name": "Nitrogen",
            "compartment": "air",
            "subcompartment": "urban air",
            "reference_unit": "kg",
            "occurrence_count": "2",
            "characterised_occurrence_count": "2",
            "tau_entry_min": "",
            "tau_entry_median": "",
            "tau_entry_max": "",
            "eta_0_95": "0",
            "loss_max_0_95": "0.1",
            "eta_0_99": "0",
            "loss_max_0_99": "0.1",
        },
        {
            "flow_id": "source-unmatched",
            "flow_name": "Unlisted flow",
            "compartment": "water",
            "subcompartment": "fresh water",
            "reference_unit": "kg",
            "occurrence_count": "1",
            "characterised_occurrence_count": "0",
            "tau_entry_min": "",
            "tau_entry_median": "",
            "tau_entry_max": "",
            "eta_0_95": "0",
            "loss_max_0_95": "0",
            "eta_0_99": "0",
            "loss_max_0_99": "0",
        },
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_snapshot(path: Path, list_name: str) -> None:
    fieldnames = [
        "source_file",
        "list_name",
        "flow_uuid",
        "flow_name",
        "context",
        "unit",
        "cas_number",
        "synonyms",
    ]
    rows = [
        {
            "source_file": f"{list_name}.csv",
            "list_name": list_name,
            "flow_uuid": "uuid-consensus",
            "flow_name": "Carbon dioxide",
            "context": "emission/air/urban air",
            "unit": "kg",
            "cas_number": "",
            "synonyms": "",
        },
        {
            "source_file": f"{list_name}.csv",
            "list_name": list_name,
            "flow_uuid": "uuid-only",
            "flow_name": "Target uuid row",
            "context": "air/urban air",
            "unit": "kg",
            "cas_number": "",
            "synonyms": "",
        },
        {
            "source_file": f"{list_name}.csv",
            "list_name": list_name,
            "flow_uuid": "uuid-methane",
            "flow_name": "Methane",
            "context": "air/urban air",
            "unit": "kg",
            "cas_number": "",
            "synonyms": "",
        },
        {
            "source_file": f"{list_name}.csv",
            "list_name": list_name,
            "flow_uuid": "uuid-conflict",
            "flow_name": "Wrong uuid target",
            "context": "air/urban air",
            "unit": "kg",
            "cas_number": "",
            "synonyms": "",
        },
        {
            "source_file": f"{list_name}.csv",
            "list_name": list_name,
            "flow_uuid": "uuid-name-match",
            "flow_name": "Flow by name",
            "context": "soil/agricultural",
            "unit": "kg",
            "cas_number": "",
            "synonyms": "",
        },
        {
            "source_file": f"{list_name}.csv",
            "list_name": list_name,
            "flow_uuid": "uuid-amb-1",
            "flow_name": "Nitrogen",
            "context": "air/urban air",
            "unit": "kg",
            "cas_number": "",
            "synonyms": "",
        },
        {
            "source_file": f"{list_name}.csv",
            "list_name": list_name,
            "flow_uuid": "uuid-amb-2",
            "flow_name": "Nitrogen",
            "context": "air/urban air",
            "unit": "kg",
            "cas_number": "",
            "synonyms": "",
        },
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_asset_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    manifest = {
        "upstream_repo_url": "https://github.com/UNEP-Economy-Division/GLAD-ElementaryFlowResources",
        "snapshot_generation_date": "2026-06-10",
        "notes": [
            "flow-list CSVs are the authoritative assets for direct target matching",
            "pairwise Excel mappings exist upstream but are not the primary matcher here",
        ],
        "targets": {},
    }
    for name in ("ecoinventEFv3.7", "ILCD_EFv3.0", "FEDEFLv1.0.3", "IDEA_EFv2.3"):
        _write_snapshot(path / f"{name}.csv", name)
        manifest["targets"][name] = {"snapshot": f"{name}.csv"}
    (path / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_match_statuses_cover_uuid_name_conflict_ambiguity_and_unmatched(tmp_path: Path) -> None:
    priority_csv = _write_priority_csv(tmp_path / "priority.csv")
    asset_dir = _write_asset_dir(tmp_path / "assets")

    dataset, pair, rows, manifest = build_priority_glad_rows(priority_csv, audit_tau=0.95, asset_dir=asset_dir)

    assert dataset.get_tau_pair(0.95).tau == 0.95
    assert load_asset_manifest(asset_dir)["upstream_repo_url"] == manifest["upstream_repo_url"]

    by_id = {item.priority_row.flow_id: item for item in rows}
    target = "ecoinventEFv3.7"
    assert by_id["uuid-consensus"].target_results[target].match_status == "exact_consensus"
    assert by_id["uuid-only"].target_results[target].match_status == "exact_uuid"
    assert by_id["source-no-uuid"].target_results[target].match_status == "exact_name_context_unit"
    assert by_id["uuid-conflict"].target_results[target].match_status == "conflict"
    assert by_id["source-ambiguous"].target_results[target].match_status == "ambiguous"
    assert by_id["source-unmatched"].target_results[target].match_status == "unmatched"
    assert classify_unresolved_row(by_id["source-no-uuid"].priority_row, pair) == "certificate_breaking"
    assert classify_unresolved_row(by_id["uuid-conflict"].priority_row, pair) == "group_risk_only"
    assert classify_unresolved_row(by_id["source-unmatched"].priority_row, pair) == "not_lcia_visible"


def test_write_outputs_creates_expected_reports_and_sorts_unresolved(tmp_path: Path) -> None:
    priority_csv = _write_priority_csv(tmp_path / "priority.csv")
    asset_dir = _write_asset_dir(tmp_path / "assets")
    output_dir = tmp_path / "out"

    summary = write_priority_glad_outputs(priority_csv, output_dir, audit_tau=0.95, asset_dir=asset_dir)

    assert Path(summary.summary_json_path).exists()
    assert Path(summary.compact_summary_csv_path).exists()
    assert summary.total_rows == 6
    assert summary.matched_all_targets == 3
    assert summary.matched_no_targets == 3
    counts = summary.target_status_counts["ecoinventEFv3.7"]
    assert counts["exact_consensus"] == 1
    assert counts["exact_uuid"] == 1
    assert counts["exact_name_context_unit"] == 1
    assert counts["conflict"] == 1
    assert counts["ambiguous"] == 1
    assert counts["unmatched"] == 1

    unresolved_rows = _read_csv_rows(Path(summary.per_target_unmatched_csv_paths["ecoinventEFv3.7"]))
    assert [row["flow_id"] for row in unresolved_rows] == [
        "uuid-conflict",
        "source-ambiguous",
        "source-unmatched",
    ]
    assert unresolved_rows[0]["unresolved_class"] == "group_risk_only"
    assert unresolved_rows[2]["unresolved_class"] == "not_lcia_visible"
    assert "candidate_rows" in unresolved_rows[0]
    assert priority_glad_output_columns(build_priority_glad_rows(priority_csv, audit_tau=0.95, asset_dir=asset_dir)[1])[-1] == "candidate_rows"


def test_summary_json_contains_cross_target_counts(tmp_path: Path) -> None:
    priority_csv = _write_priority_csv(tmp_path / "priority.csv")
    asset_dir = _write_asset_dir(tmp_path / "assets")
    output_dir = tmp_path / "out"

    summary = write_priority_glad_outputs(priority_csv, output_dir, audit_tau=0.95, asset_dir=asset_dir)
    payload = json.loads(Path(summary.summary_json_path).read_text(encoding="utf-8"))

    assert payload["matched_all_targets"] == 3
    assert payload["matched_no_targets"] == 3
    assert payload["matched_some_targets"] == 0
    assert payload["target_status_counts"]["IDEA_EFv2.3"]["ambiguous"] == 1
    assert payload["top_unresolved_by_target"]["FEDEFLv1.0.3"][0]["flow_id"] == "uuid-conflict"
