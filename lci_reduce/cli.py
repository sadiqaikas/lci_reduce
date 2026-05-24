"""Command-line interface."""

from __future__ import annotations

import argparse
import json
from typing import Any, Optional, Sequence

from .cf_resolution import PromptCallback
from .errors import LciReduceError
from .flow_priority import create_flow_priority
from .jsonld_reader import index_archive, inspect_archives
from .jsonld_writer import ProgressCallback, create_lite_database
from .models import CreateConfig, FlowPriorityConfig
from .priority_analyser import (
    PriorityRecord,
    build_priority_summary,
    load_priority_dataset,
    match_flow_ids,
    match_flow_names,
    parse_repeated_option_items,
    SelectionMatchResult,
    write_ranked_csv,
    write_summary_json,
)


CLI_GUIDE_SECTIONS = [
    {
        "title": "1. Inspect Before You Run",
        "summary": (
            "Use `inspect` first. It tells you whether the database already contains LCIA methods, "
            "how many categories are available, and whether you need an external methods archive."
        ),
        "command": (
            "lci_reduce inspect \\\n"
            "  --database /path/to/original_database.zip \\\n"
            "  --methods /path/to/methods.zip"
        ),
        "notes": [
            "If the database already contains impact methods, you can leave `--methods` empty for later commands.",
            "The command is read-only. It does not create a lite database and does not modify the input archive.",
            "JSON output is printed to stdout so you can redirect it into a file if you want a record of the scan.",
        ],
    },
    {
        "title": "2. Create A Lite Database",
        "summary": (
            "Run deterministic signed tau-cover reduction. This is the command that writes the reduced JSON-LD ZIP "
            "plus manifests, validation, warnings, config, and report artefacts."
        ),
        "command": (
            "lci_reduce create \\\n"
            "  --database /path/to/original_database.zip \\\n"
            "  --methods /path/to/methods.zip \\\n"
            "  --output /path/to/out_dir \\\n"
            "  --tau 0.95 \\\n"
            "  --method-selection all \\\n"
            "  --uncharacterised-policy keep \\\n"
            "  --strict-units true"
        ),
        "notes": [
            "Use `--method-selection all` to include every detected LCIA category, or pass your existing selection string if you already use a narrower set.",
            "Strict units are on by default. If unit compatibility is ambiguous, the run should fail instead of guessing.",
            "Expected outputs are the lite database ZIP plus `exchange_manifest.csv`, `process_manifest.csv`, `run_summary.json`, warnings, config, and the short PDF report.",
        ],
    },
    {
        "title": "3. Generate LCIA Flow Priority Sidecars",
        "summary": (
            "This sidecar workflow ranks elementary flows by LCIA-criticality for transfer and mapping-repair work. "
            "It does not rewrite the database and does not create a lite ZIP."
        ),
        "command": (
            "lci_reduce priority \\\n"
            "  --database /path/to/original_database.zip \\\n"
            "  --methods /path/to/methods.zip \\\n"
            "  --output /path/to/out_dir \\\n"
            "  --method-selection all \\\n"
            "  --audit-tau 0.95 0.99 \\\n"
            "  --strict-units true"
        ),
        "notes": [
            "Expected outputs are `lcia_flow_priority.csv` and `lcia_flow_priority_metadata.json`.",
            "The CSV is a compact screening file: `eta` is exact for single-flow omission, while grouped flow analysis needs conservative bounds unless a future ledger mode is used.",
            "Use audit tau values such as `0.95 0.99` to emit multiple `eta_<tau>` and `loss_max_<tau>` column pairs in one file.",
        ],
    },
    {
        "title": "4. Analyse An Existing Priority File",
        "summary": (
            "Use the compact priority CSV directly. This analyser does not need the original database ZIP, "
            "does not need a failed mappings CSV, and does not rewrite anything."
        ),
        "command": (
            "lci_reduce analyse-priority \\\n"
            "  --priority-csv /path/to/lcia_flow_priority.csv \\\n"
            "  --metadata-json /path/to/lcia_flow_priority_metadata.json \\\n"
            "  --audit-tau 0.95 \\\n"
            "  --top-n 20 \\\n"
            "  --select-flow-id flow-1,flow-2 \\\n"
            "  --select-flow-name \"Sulfur dioxide\" \\\n"
            "  --select-flow-name \"Methane, fossil\" \\\n"
            "  --output-ranked-csv /path/to/ranked.csv \\\n"
            "  --output-summary-json /path/to/priority_analysis_summary.json"
        ),
        "notes": [
            "The analyser validates the CSV schema and detects available audit tau column pairs before doing any ranking.",
            "For grouped selected flows it reports a compact-screen bound only: `max eta <= exact group eta <= min(tau, sum loss_max)`.",
            "If a flow name contains commas, prefer repeating `--select-flow-name` instead of packing several names into one comma-separated argument.",
        ],
    },
    {
        "title": "5. Start The Desktop GUI",
        "summary": (
            "Use the GUI when you want guided file picking, embedded method hints, interactive priority screening, "
            "or reduction-curve comparisons."
        ),
        "command": "lci_reduce-gui",
        "notes": [
            "The GUI uses the same backend as the CLI, so CLI and GUI results come from the same reduction logic.",
            "If the console script is not installed, `python -m lci_reduce gui` and `python main.py` remain available.",
        ],
    },
    {
        "title": "6. Alternate Entrypoints",
        "summary": "These forms are useful when you are running directly from a checkout without installing console scripts.",
        "command": (
            "python -m lci_reduce cli inspect --database /path/to/original_database.zip\n"
            "python -m lci_reduce gui\n"
            "python main.py cli analyse-priority --priority-csv /path/to/lcia_flow_priority.csv\n"
            "python main.py"
        ),
        "notes": [
            "`python -m lci_reduce` defaults to the GUI when you omit `cli` or `gui`.",
            "`python main.py cli ...` is equivalent to the installed `lci_reduce ...` command form.",
        ],
    },
]


def _bool_arg(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Expected true or false")


def _positive_int_arg(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Expected a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lci_reduce",
        description=(
            "ZIP-only openLCA JSON-LD reduction and LCIA priority tooling. "
            "Use `inspect` before `create` or `priority`, and use `analyse-priority` "
            "to screen an already generated lcia_flow_priority.csv."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser(
        "inspect",
        help="Inspect database and optional methods inputs without writing outputs.",
    )
    inspect_parser.add_argument("--database", required=True)
    inspect_parser.add_argument("--methods")

    create_parser = subparsers.add_parser(
        "create",
        help="Create a lite JSON-LD database ZIP plus validation artefacts.",
    )
    create_parser.add_argument("--database", required=True)
    create_parser.add_argument("--methods")
    create_parser.add_argument("--output", required=True)
    create_parser.add_argument("--tau", required=True, type=float)
    create_parser.add_argument("--method-selection", required=True)
    create_parser.add_argument(
        "--uncharacterised-policy",
        default="drop",
        choices=["keep", "drop", "fail"],
    )
    create_parser.add_argument("--strict-units", default=True, type=_bool_arg)
    create_parser.add_argument("--tolerance", default=1e-12, type=float)
    create_parser.add_argument("--cf-resolution-file")

    priority_parser = subparsers.add_parser(
        "priority",
        help="Generate LCIA flow-priority sidecars without rewriting the database.",
    )
    priority_parser.add_argument("--database", required=True)
    priority_parser.add_argument("--methods")
    priority_parser.add_argument("--output", required=True)
    priority_parser.add_argument("--method-selection", required=True)
    priority_parser.add_argument("--audit-tau", nargs="+", default=[0.95, 0.99], type=float)
    priority_parser.add_argument("--strict-units", default=True, type=_bool_arg)
    priority_parser.add_argument("--tolerance", default=1e-12, type=float)
    priority_parser.add_argument("--cf-resolution-file")

    analyse_priority_parser = subparsers.add_parser(
        "analyse-priority",
        help="Analyse an existing lcia_flow_priority.csv without loading the original database.",
    )
    analyse_priority_parser.add_argument("--priority-csv", required=True)
    analyse_priority_parser.add_argument("--metadata-json")
    analyse_priority_parser.add_argument(
        "--audit-tau",
        type=float,
        default=None,
        help="Audit tau to analyse. Defaults to 0.95 if present, otherwise the first detected tau pair.",
    )
    analyse_priority_parser.add_argument("--top-n", type=_positive_int_arg, default=20)
    analyse_priority_parser.add_argument("--select-flow-id", action="append")
    analyse_priority_parser.add_argument("--select-flow-name", action="append")
    analyse_priority_parser.add_argument("--output-summary-json")
    analyse_priority_parser.add_argument("--output-ranked-csv")
    return parser


def inspect_command(database: str, methods: Optional[str]) -> dict:
    database_archive = index_archive(
        database,
        require_processes=True,
        require_flows=True,
        conversion_scope="inspect",
    )
    methods_archive = (
        index_archive(
            methods,
            require_processes=False,
            require_flows=False,
            conversion_scope="inspect",
        )
        if methods
        else None
    )
    result = inspect_archives(database_archive, methods_archive)
    return {
        "detected_processes": result.n_processes,
        "detected_flows": result.n_flows,
        "detected_elementary_flows": result.n_elementary_flows,
        "detected_lcia_methods": result.n_methods,
        "detected_lcia_categories": result.n_categories,
        "database_contains_impact_methods": result.database_contains_impact_methods,
        "database_lcia_methods": result.database_methods,
        "database_lcia_categories": result.database_categories,
        "database_methods_hint": (
            "Database already contains impact methods. You can leave --methods empty and use the database methods."
            if result.database_contains_impact_methods
            else "Database does not contain impact methods. Provide --methods to use external LCIA methods."
        ),
        "possible_method_family_names": result.family_names,
        "cf_mappings_buildable": result.cf_mappings_buildable,
        "categories_with_factors": result.categories_with_factors,
        "method_names": result.method_names,
        "lcia_categories": [row.__dict__ for row in result.lcia_categories],
        "empty_lcia_categories": [row.__dict__ for row in result.empty_lcia_categories],
    }


def create_command(
    database: str,
    methods: Optional[str],
    output: str,
    tau: float,
    method_selection: str,
    uncharacterised_policy: str,
    strict_units: bool,
    tolerance: float,
    cf_resolution_file: Optional[str] = None,
    *,
    cf_prompt: Optional[PromptCallback] = None,
    progress_callback: Optional[ProgressCallback] = None,
):
    config = CreateConfig(
        database=database,
        methods=methods,
        output_dir=output,
        tau=tau,
        method_selection=method_selection,
        uncharacterised_policy=uncharacterised_policy,
        strict_units=strict_units,
        tolerance=tolerance,
        cf_resolution_file=cf_resolution_file,
    )
    return create_lite_database(config, cf_prompt=cf_prompt, progress_callback=progress_callback)


def priority_command(
    database: str,
    methods: Optional[str],
    output: str,
    method_selection: str,
    audit_tau: list[float],
    strict_units: bool,
    tolerance: float,
    cf_resolution_file: Optional[str] = None,
    *,
    cf_prompt: Optional[PromptCallback] = None,
    progress_callback: Optional[ProgressCallback] = None,
):
    config = FlowPriorityConfig(
        database=database,
        methods=methods,
        output_dir=output,
        method_selection=method_selection,
        audit_tau_values=list(audit_tau),
        strict_units=strict_units,
        tolerance=tolerance,
        cf_resolution_file=cf_resolution_file,
    )
    return create_flow_priority(config, cf_prompt=cf_prompt, progress_callback=progress_callback)


def analyse_priority_command(
    priority_csv: str,
    metadata_json: Optional[str] = None,
    audit_tau: float | None = None,
    top_n: int = 20,
    select_flow_id: Sequence[str] | None = None,
    select_flow_name: Sequence[str] | None = None,
    output_summary_json: Optional[str] = None,
    output_ranked_csv: Optional[str] = None,
) -> dict[str, Any]:
    dataset = load_priority_dataset(priority_csv, metadata_json)
    pair = dataset.get_tau_pair(audit_tau)
    flow_id_items = parse_repeated_option_items(select_flow_id)
    flow_name_items = parse_repeated_option_items(select_flow_name)

    selection_match = None
    selected_rows = None
    if flow_id_items or flow_name_items:
        id_matches = match_flow_ids(dataset, flow_id_items)
        name_matches = match_flow_names(dataset, flow_name_items)
        selection_match = _combine_selection_results(id_matches, name_matches)
        selected_rows = selection_match.matched_rows

    summary = build_priority_summary(
        dataset,
        pair,
        top_n=top_n,
        selected_rows=selected_rows,
        selection_match=selection_match,
    )

    if output_ranked_csv:
        write_ranked_csv(output_ranked_csv, dataset.rows, pair)
    if output_summary_json:
        write_summary_json(output_summary_json, summary)
    return summary


def _combine_selection_results(*results: SelectionMatchResult) -> SelectionMatchResult:
    matched_rows: list[PriorityRecord] = []
    matched_flow_ids: list[str] = []
    unmatched_items: list[str] = []
    ambiguous_items: dict[str, list[str]] = {}
    seen_flow_ids: set[str] = set()
    for result in results:
        for row in result.matched_rows:
            if row.flow_id in seen_flow_ids:
                continue
            seen_flow_ids.add(row.flow_id)
            matched_rows.append(row)
            matched_flow_ids.append(row.flow_id)
        unmatched_items.extend(result.unmatched_items)
        ambiguous_items.update(result.ambiguous_items)
    return SelectionMatchResult(
        matched_rows=matched_rows,
        matched_flow_ids=matched_flow_ids,
        unmatched_items=unmatched_items,
        ambiguous_items=ambiguous_items,
    )


def format_analyse_priority_report(summary: dict[str, Any]) -> str:
    overview = summary["overview"]
    selected_tau = summary["selected_audit_tau"]
    selected_columns = summary["selected_columns"]
    lines = [
        "Priority File Analyser",
        f"Priority CSV: {summary['priority_csv']}",
        f"Metadata JSON: {summary.get('metadata_json') or '-'}",
        (
            "Schema validation: ok | detected tau values: "
            + ", ".join(format(float(value), ".15g") for value in summary["schema_validation"]["detected_tau_values"])
        ),
        f"Selected tau: {format(float(selected_tau), '.15g')} "
        f"({selected_columns['eta_column']} / {selected_columns['loss_max_column']})",
        "",
        "Overview",
        f"  total flows: {overview['total_flows']}",
        f"  characterised flows: {overview['characterised_flows']}",
        f"  partly characterised flows: {overview['partly_characterised_flows']}",
        f"  uncharacterised flows: {overview['uncharacterised_flows']}",
    ]
    for token, metrics in sorted(overview["tau_metrics"].items(), key=lambda item: item[1]["tau"]):
        tau_text = format(float(metrics["tau"]), ".15g")
        lines.append(
            f"  tau {tau_text}: eta>0 flows={metrics['eta_positive_count']} | "
            f"loss_max>0 flows={metrics['loss_positive_count']}"
        )

    lines.extend(
        [
            "",
            f"Top {summary['top_n']} By Eta",
        ]
    )
    for row in summary["top_by_eta"]:
        lines.append(_format_ranked_row(row))

    lines.extend(
        [
            "",
            f"Top {summary['top_n']} By Loss Max",
        ]
    )
    for row in summary["top_by_loss_max"]:
        lines.append(_format_ranked_row(row))

    selection_match = summary.get("selection_match")
    selected_analysis = summary.get("selected_flow_analysis")
    if selection_match is not None:
        lines.extend(
            [
                "",
                "Selected Flows",
                f"  matched flows: {selection_match['matched_count']}",
                f"  unmatched items: {len(selection_match['unmatched_items'])}",
                f"  ambiguous items: {len(selection_match['ambiguous_items'])}",
            ]
        )
        matched_flows = selection_match.get("matched_flows") or []
        for row in matched_flows:
            lines.append(
                "  "
                + _format_ranked_row({**row, "rank": row.get("rank", "")}, include_rank=False)
            )
        if selection_match["unmatched_items"]:
            lines.append("  unmatched: " + ", ".join(selection_match["unmatched_items"]))
        if selection_match["ambiguous_items"]:
            for item, candidates in selection_match["ambiguous_items"].items():
                lines.append(f"  ambiguous {item}: " + "; ".join(candidates))

    if selected_analysis is not None:
        lines.extend(
            [
                "",
                "Compact-Screen Group Bound",
                f"  lower bound: {_format_metric(selected_analysis['lower_bound_eta'])}",
                f"  upper bound: {_format_metric(selected_analysis['upper_bound_eta'])}",
                f"  sum loss_max: {_format_metric(selected_analysis['sum_loss_max'])}",
                f"  capped at tau: {'yes' if selected_analysis['capped_at_tau'] else 'no'}",
                f"  interval: {selected_analysis['interval']}",
            ]
        )
        for message in selected_analysis["interpretation"]:
            lines.append(f"  interpretation: {message}")
    return "\n".join(lines)


def _format_metric(value: Any) -> str:
    if value is None:
        return "-"
    return format(float(value), ".6g")


def _format_ranked_row(row: dict[str, Any], *, include_rank: bool = True) -> str:
    prefix = f"  {row['rank']:>2}. " if include_rank and row.get("rank") not in ("", None) else "  - "
    return (
        f"{prefix}{row['flow_id']} | {row['flow_name']} | "
        f"eta={_format_metric(row['eta_tau'])} | loss_max={_format_metric(row['loss_max_tau'])} | "
        f"tau_entry_min={_format_metric(row['tau_entry_min'])} | cf_status={row['cf_status']}"
    )


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            result = inspect_command(args.database, args.methods)
            print(json.dumps(result, indent=2, ensure_ascii=True))
        elif args.command == "create":
            result = create_command(
                database=args.database,
                methods=args.methods,
                output=args.output,
                tau=args.tau,
                method_selection=args.method_selection,
                uncharacterised_policy=args.uncharacterised_policy,
                strict_units=args.strict_units,
                tolerance=args.tolerance,
                cf_resolution_file=args.cf_resolution_file,
            )
            print(
                json.dumps(
                    {
                        "output_zip": result.output_zip,
                        "exchange_manifest_csv": result.exchange_manifest_csv,
                        "process_manifest_csv": result.process_manifest_csv,
                        "run_summary_json": result.run_summary_json,
                    },
                    indent=2,
                    ensure_ascii=True,
                )
            )
        elif args.command == "priority":
            result = priority_command(
                database=args.database,
                methods=args.methods,
                output=args.output,
                method_selection=args.method_selection,
                audit_tau=args.audit_tau,
                strict_units=args.strict_units,
                tolerance=args.tolerance,
                cf_resolution_file=args.cf_resolution_file,
            )
            print(
                json.dumps(
                    {
                        "flow_priority_csv": result.flow_priority_csv,
                        "flow_priority_metadata_json": result.flow_priority_metadata_json,
                    },
                    indent=2,
                    ensure_ascii=True,
                )
            )
        elif args.command == "analyse-priority":
            summary = analyse_priority_command(
                priority_csv=args.priority_csv,
                metadata_json=args.metadata_json,
                audit_tau=args.audit_tau,
                top_n=args.top_n,
                select_flow_id=args.select_flow_id,
                select_flow_name=args.select_flow_name,
                output_summary_json=args.output_summary_json,
                output_ranked_csv=args.output_ranked_csv,
            )
            print(format_analyse_priority_report(summary))
        return 0
    except LciReduceError as exc:
        parser.exit(1, f"{exc}\n")
    except Exception as exc:  # pragma: no cover - CLI guard
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
