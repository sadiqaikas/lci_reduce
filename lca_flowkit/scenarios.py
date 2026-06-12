"""Contribution-matrix construction with explicit CF ambiguity handling.

This module is where parsed exchanges become the characterised matrix that the
rest of the package can reduce or audit.

The scientific job here is to build rows of the form:

``A[row, exchange] = exchange_amount * CF * unit_conversion``

but without silently picking one characterisation factor whenever the source
data leaves more than one admissible interpretation.

The package distinguishes three cases:

``exact``
    one scientifically admissible characterised value.

``finite_cf``
    a small non-regional ambiguity, such as multiple admissible factors with
    the same units but different values.  These are expanded explicitly across
    scenario rows.

``regional_cf``
    location-driven ambiguity.  The package does not select one region
    silently; it always builds explicit shared regional scenario rows.

The resulting matrix is therefore not merely "LCIA factors applied to
exchanges".  It is a transparent encoding of both characterised contributions
and unresolved modelling ambiguity.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np

from .errors import ScenarioExpansionError, UnitResolutionError
from .models import ContributionBundle, Exchange, FlowDefinition, ImpactCategory, ImpactFactorCandidate, ProcessRecord, ScenarioRow, ScenarioStats, WarningRecord
from .units import resolve_unit_conversion
from .utils import fold_text, normalise_text, unique_preserve_order


@dataclass(frozen=True)
class _PreparedOption:
    """One admissible characterised contribution for one exchange/candidate pair.

    By the time an option reaches this dataclass, the code has already checked:

    - flow identity compatibility;
    - compartment/subcompartment compatibility; and
    - unit compatibility or conversion.

    What remains is one candidate contribution plus the location metadata that
    controls whether it is exact, finite ambiguous, or regionally ambiguous.
    """
    candidate: ImpactFactorCandidate
    contribution: float
    location_label: str
    location_tokens: tuple[str, ...]
    coverage_tokens: tuple[str, ...]


@dataclass(frozen=True)
class _Case:
    """One exchange column together with the admissible options that remain.

    A ``_Case`` is created only when an exchange could not be reduced to one
    exact characterised contribution.  Downstream helpers then expand it either
    cartesianly (finite ambiguity) or through shared regional rows.
    """
    column_index: int
    exchange: Exchange
    kind: str
    options: tuple[_PreparedOption, ...]


def _flow_matches_candidate(flow: FlowDefinition, candidate: ImpactFactorCandidate) -> bool:
    """Filter factor candidates by compartment and subcompartment compatibility.

    Flow ID matching is performed before this function is called.  This helper
    adds the narrower ecological check that the compartment labels still agree.
    The package treats compartment mismatches as "not the same environmental
    flow", even if some upstream export happened to reuse an identifier.
    """
    if candidate.compartment and fold_text(flow.compartment) != fold_text(candidate.compartment):
        return False
    if candidate.subcompartment and fold_text(flow.subcompartment) != fold_text(candidate.subcompartment):
        return False
    return True


def _candidate_coverage_tokens(candidate: ImpactFactorCandidate) -> tuple[str, ...]:
    """Extract explicit authorised-parent coverage tokens from raw factor data.

    Some sources encode that a factor applies to a parent region or to a named
    set of child regions.  We preserve those tokens separately from the factor's
    own location label so later logic can distinguish:

    - exact location match;
    - authorised parent-region coverage; and
    - fully generic fallback.
    """
    raw = candidate.raw or {}
    values = raw.get("locationCoverage") or raw.get("location_coverage") or raw.get("coveredLocations") or raw.get("covered_locations")
    if not isinstance(values, list):
        return ()
    tokens: list[str] = []
    for item in values:
        if isinstance(item, str):
            token = normalise_text(item)
            if token and token not in tokens:
                tokens.append(token)
        elif isinstance(item, dict):
            for key in ("name", "code", "region", "@id", "id"):
                value = item.get(key)
                if isinstance(value, str):
                    token = normalise_text(value)
                    if token and token not in tokens:
                        tokens.append(token)
    return tuple(tokens)


def _collapse_options(options: list[_PreparedOption]) -> tuple[tuple[_PreparedOption, ...], int]:
    """Collapse scientifically equivalent options before scenario expansion.

    This keeps the scenario space readable and avoids multiplying rows that are
    scientifically indistinguishable after unit conversion and location
    interpretation.

    We collapse only when the candidate metadata relevant to scientific meaning
    is the same after normalisation.  Two candidates that merely happen to
    produce the same post-conversion contribution must remain distinct if they
    differ in unit basis, flow property, source, location semantics, or other
    materially relevant metadata.
    """
    deduped: dict[tuple[object, ...], _PreparedOption] = {}
    collapsed = 0
    for option in options:
        key = (
            option.candidate.category_id,
            option.candidate.method_id,
            option.candidate.flow_id,
            format(option.candidate.value, ".15g"),
            option.candidate.unit_id,
            fold_text(option.candidate.unit_name),
            option.candidate.flow_property_id,
            fold_text(option.candidate.flow_property_name),
            option.candidate.location.location_id,
            normalise_text(option.candidate.location.name),
            normalise_text(option.candidate.location.region),
            fold_text(option.candidate.compartment),
            fold_text(option.candidate.subcompartment),
            option.location_label,
            option.location_tokens,
            option.coverage_tokens,
            option.candidate.source_path,
        )
        if key in deduped:
            collapsed += 1
            continue
        deduped[key] = option
    return tuple(deduped.values()), collapsed


def _prepare_options(
    exchange: Exchange,
    flow: FlowDefinition,
    category: ImpactCategory,
    unit_registry: dict[str, object],
    *,
    strict_units: bool,
) -> tuple[tuple[_PreparedOption, ...], list[WarningRecord]]:
    """Resolve unit-compatible factor candidates into characterised options.

    The output options are the admissible values of:

    ``amount(exchange) * CF(candidate) * unit_conversion(exchange -> factor)``

    for one process exchange under one impact category.

    This stage does *not* choose among ambiguous factors.  Its role is to:

    1. gather all factor candidates attached to the same flow;
    2. remove compartment-incompatible candidates;
    3. apply strict unit filtering and conversion;
    4. convert surviving candidates into characterised contribution options;
    5. collapse only metadata-equivalent duplicates.

    If strict unit handling is enabled and every candidate fails unit
    compatibility, the function raises immediately so the caller does not
    mistake a physical incompatibility for "the flow is simply uncharacterised".
    """
    candidates = category.factors_by_flow.get(flow.flow_id, [])
    warnings: list[WarningRecord] = []
    if not candidates:
        return (), warnings
    prepared: list[_PreparedOption] = []
    unit_failures: list[str] = []
    for candidate in candidates:
        if not _flow_matches_candidate(flow, candidate):
            continue
        try:
            unit_result = resolve_unit_conversion(
                exchange,
                flow,
                candidate,
                unit_registry,  # type: ignore[arg-type]
                strict_units=strict_units,
            )
        except UnitResolutionError as exc:
            unit_failures.append(str(exc))
            continue
        location_tokens = candidate.location.tokens()
        # The contribution stored here is already in the factor's reference
        # basis, so later scenario expansion can stay purely combinatorial.
        prepared.append(
            _PreparedOption(
                candidate=candidate,
                contribution=float(exchange.amount) * float(candidate.value) * float(unit_result.conversion_factor),
                location_label=candidate.location.name or candidate.location.location_id or candidate.location.region,
                location_tokens=location_tokens,
                coverage_tokens=_candidate_coverage_tokens(candidate),
            )
        )
    if prepared:
        deduped, collapsed = _collapse_options(prepared)
        if collapsed:
            warnings.append(
                WarningRecord(
                    code="duplicate_cf_collapse",
                    message=f"Collapsed {collapsed} metadata-equivalent CF candidate(s)",
                    process_id="",
                    flow_id=flow.flow_id,
                    flow_name=flow.name,
                    category_id=category.category_id,
                    category_name=category.name,
                )
            )
        return deduped, warnings
    if unit_failures and strict_units:
        raise UnitResolutionError(unit_failures[0])
    return (), warnings


def _options_for_location(case: _Case, location_label: str) -> tuple[str, tuple[_PreparedOption, ...]]:
    """Resolve which options are admissible for one shared regional scenario.

    The package avoids a naive per-exchange location cross-product.  Instead it
    iterates over shared location labels and asks, for each ambiguous exchange,
    which options remain admissible under that label.

    Priority of admissibility:

    - exact location token match;
    - authorised parent-region coverage;
    - generic blank-location fallback;
    - no applicable option.
    """
    exact = tuple(option for option in case.options if location_label and location_label in option.location_tokens)
    if exact:
        return "exact_location", exact
    parent = tuple(option for option in case.options if location_label and location_label in option.coverage_tokens)
    if parent:
        smallest = min(len(option.coverage_tokens) for option in parent if option.coverage_tokens)
        return "authorised_parent_region", tuple(
            option for option in parent if len(option.coverage_tokens) == smallest
        )
    generic = tuple(option for option in case.options if not option.location_tokens and not option.coverage_tokens)
    if generic:
        return "generic_fallback", generic
    return "not_applicable", ()


def _cartesian_rows(base_rows: list[tuple[np.ndarray, str, str]], case: _Case, n_cols: int) -> list[tuple[np.ndarray, str, str]]:
    """Expand one finite non-regional ambiguity case across the current rows.

    This is used only for finite, non-regional ambiguity.  If one exchange has
    ``k`` admissible factor values and no regional semantics, every existing row
    branches into ``k`` rows, each carrying one of those values in the exchange
    column.
    """
    if not base_rows:
        base_rows = [(np.zeros(n_cols, dtype=float), "exact", "exact")]
    rows: list[tuple[np.ndarray, str, str]] = []
    for vector, scenario_type, label in base_rows:
        for option in case.options:
            new_vector = vector.copy()
            new_vector[case.column_index] += option.contribution
            rows.append((new_vector, scenario_type, f"{label} | exchange:{case.exchange.exchange_id} | cf={format(option.candidate.value, '.15g')}"))
    return rows


def _regional_rows(regional_cases: list[_Case], n_cols: int) -> list[tuple[np.ndarray, str, str]]:
    """Build shared-location scenario rows without a full location cross-product.

    The obvious but expensive approach would be:

    - for each ambiguous exchange, list all admissible regional options;
    - take the full cartesian product across exchanges.

    That becomes combinatorially large very quickly.  The cleaner scientific
    approximation used here groups by shared location labels first, then expands
    only the options admissible under that shared label.  This keeps the rows
    explicit without pretending that the full regional joint distribution is
    known.
    """
    location_labels = unique_preserve_order(
        label
        for case in regional_cases
        for option in case.options
        for label in (
            option.location_label,
            *option.coverage_tokens,
        )
        if label
    )
    if not location_labels:
        location_labels = ["regional_unlocated"]
    rows: list[tuple[np.ndarray, str, str]] = []
    for location_label in location_labels:
        per_case_options: list[list[_PreparedOption | None]] = []
        notes: list[str] = [f"location={location_label}"]
        for case in regional_cases:
            reason, options = _options_for_location(case, location_label)
            notes.append(f"exchange:{case.exchange.exchange_id}:{reason}")
            per_case_options.append(list(options) if options else [None])
        for combo in product(*per_case_options):
            vector = np.zeros(n_cols, dtype=float)
            combo_notes = list(notes)
            for case_index, option in enumerate(combo):
                if option is None:
                    continue
                vector[regional_cases[case_index].column_index] += option.contribution
                combo_notes.append(f"cf={format(option.candidate.value, '.15g')}")
            rows.append((vector, "regional_cf", " | ".join(combo_notes)))
    return rows


def _build_category_rows(
    candidate_columns: list[Exchange],
    flow_lookup: dict[str, FlowDefinition],
    category: ImpactCategory,
    unit_registry: dict[str, object],
    *,
    strict_units: bool,
    max_rows_per_process: int,
) -> tuple[list[tuple[np.ndarray, ScenarioRow]], set[int], list[WarningRecord], ScenarioStats]:
    """Build all coverage rows for one process/category combination.

    A "coverage row" is one characterised LCIA row that the reducer must cover.
    If a category has no unresolved ambiguity in this process, it contributes
    exactly one row.  If admissible CF ambiguity remains, it contributes
    multiple rows so the later tau-cover must satisfy *all* of them.

    This is the key modelling decision that prevents the clean package from
    silently committing to one CF interpretation:

    - exact contributions are shared by every row;
    - finite ambiguity is expanded explicitly;
    - regional ambiguity always becomes explicit regional scenario rows.
    """
    n_cols = len(candidate_columns)
    exact_vector = np.zeros(n_cols, dtype=float)
    finite_cases: list[_Case] = []
    regional_cases: list[_Case] = []
    characterised_columns: set[int] = set()
    warnings: list[WarningRecord] = []
    stats = ScenarioStats()

    # Each candidate column represents one elementary exchange occurrence in the
    # process being audited or reduced.
    for column_index, exchange in enumerate(candidate_columns):
        flow = flow_lookup.get(exchange.flow_id)
        if flow is None:
            continue
        options, option_warnings = _prepare_options(exchange, flow, category, unit_registry, strict_units=strict_units)
        warnings.extend(option_warnings)
        if not options:
            continue
        characterised_columns.add(column_index)
        # "Regional structure" means the ambiguity is about admissible
        # locations, not merely about multiple factor values.
        has_regional_structure = any(option.location_tokens or option.coverage_tokens for option in options)
        if len(options) == 1 and not has_regional_structure:
            exact_vector[column_index] += options[0].contribution
            continue
        if has_regional_structure:
            regional_cases.append(_Case(column_index=column_index, exchange=exchange, kind="regional_cf", options=options))
        else:
            finite_cases.append(_Case(column_index=column_index, exchange=exchange, kind="finite_cf", options=options))

    # Exact contributions are shared by every scenario row for the category.
    base_rows: list[tuple[np.ndarray, str, str]] = [(exact_vector, "exact", "exact")]
    for case in finite_cases:
        base_rows = _cartesian_rows(base_rows, case, n_cols)
        if len(base_rows) > max_rows_per_process:
            raise ScenarioExpansionError(
                f"Scenario expansion exceeded max_rows_per_process={max_rows_per_process} while processing exchange:{case.exchange.exchange_id}"
            )
    if regional_cases:
        regional_rows = _regional_rows(regional_cases, n_cols)
        if len(regional_rows) * len(base_rows) > max_rows_per_process:
            raise ScenarioExpansionError(
                f"Scenario expansion exceeded max_rows_per_process={max_rows_per_process} while processing regional CF rows for category:{category.category_id}"
            )
        combined: list[tuple[np.ndarray, ScenarioRow]] = []
        for base_vector, _base_type, base_label in base_rows:
            for index, (extra_vector, scenario_type, scenario_label) in enumerate(regional_rows):
                vector = base_vector + extra_vector
                combined.append(
                    (
                        vector,
                        ScenarioRow(
                            row_id=f"{category.category_id}:regional:{index}",
                            category_id=category.category_id,
                            category_name=category.name,
                            method_id=category.method_id,
                            method_name=category.method_name,
                            scenario_type=scenario_type,
                            scenario_label=scenario_label if base_label == "exact" else f"{base_label} | {scenario_label}",
                        ),
                    )
                )
        stats.regional_rows += len(combined)
        return combined, characterised_columns, warnings, stats

    rows: list[tuple[np.ndarray, ScenarioRow]] = []
    for index, (vector, scenario_type, label) in enumerate(base_rows):
        row_type = "exact" if not finite_cases else "finite_cf"
        rows.append(
            (
                vector,
                ScenarioRow(
                    row_id=f"{category.category_id}:{index}",
                    category_id=category.category_id,
                    category_name=category.name,
                    method_id=category.method_id,
                    method_name=category.method_name,
                    scenario_type=row_type if scenario_type == "exact" else scenario_type,
                    scenario_label=label,
                ),
            )
        )
    if finite_cases:
        stats.finite_rows += len(rows)
    else:
        stats.exact_rows += len(rows)
    return rows, characterised_columns, warnings, stats


def build_process_contribution_bundle(
    process: ProcessRecord,
    flow_lookup: dict[str, FlowDefinition],
    categories: list[ImpactCategory],
    unit_registry: dict[str, object],
    *,
    strict_units: bool,
    max_rows_per_process: int = 300000,
) -> tuple[ContributionBundle, list[WarningRecord]]:
    """Build the full characterised contribution bundle for one process.

    The returned bundle is the single source of truth for both public
    workflows:

    - the reducer consumes ``bundle.matrix`` to choose retained exchanges;
    - the priority generator consumes the same matrix to audit entry thresholds,
      ``eta``, and ``loss_max``.

    That shared construction path is important scientifically.  It means the
    compact priority metrics are derived from the *same* characterised
    contribution rows that justify the reduction certificate.

    Protection policy:

    - provider-linked exchanges are protected;
    - quantitative-reference exchanges are protected;
    - uncharacterised elementary exchanges are protected by default.
    """
    # Candidate columns are elementary exchanges only.  Other exchange families
    # are preserved by policy and are therefore excluded from the characterised
    # matrix entirely.
    candidate_columns = [
        exchange
        for exchange in process.exchanges
        if exchange.flow_id in flow_lookup and flow_lookup[exchange.flow_id].is_elementary
    ]
    matrix_rows: list[np.ndarray] = []
    row_metadata: list[ScenarioRow] = []
    warnings: list[WarningRecord] = []
    characterised_any = [False] * len(candidate_columns)
    stats = ScenarioStats()

    for category in categories:
        rows, characterised_columns, category_warnings, category_stats = _build_category_rows(
            candidate_columns,
            flow_lookup,
            category,
            unit_registry,
            strict_units=strict_units,
            max_rows_per_process=max_rows_per_process,
        )
        warnings.extend(category_warnings)
        stats.category_rows += len(rows)
        stats.exact_rows += category_stats.exact_rows
        stats.finite_rows += category_stats.finite_rows
        stats.regional_rows += category_stats.regional_rows
        for column_index in characterised_columns:
            characterised_any[column_index] = True
        for vector, metadata in rows:
            if np.any(np.abs(vector) > 1e-12):
                matrix_rows.append(vector)
                row_metadata.append(metadata)

    # ``characterised_any[index]`` tracks whether at least one selected impact
    # category supplied a usable characterised contribution for that exchange.
    # Exchanges with no characterised contribution remain protected so the
    # reducer does not silently discard inventory that the chosen methods do not
    # see.
    protected = [
        bool(exchange.provider_id or exchange.is_quantitative_reference or not characterised_any[index])
        for index, exchange in enumerate(candidate_columns)
    ]
    bundle = ContributionBundle(
        matrix=np.vstack(matrix_rows) if matrix_rows else np.zeros((0, len(candidate_columns)), dtype=float),
        candidate_indices=[exchange.index for exchange in candidate_columns],
        exchange_keys=[exchange.exchange_id for exchange in candidate_columns],
        flow_ids=[exchange.flow_id for exchange in candidate_columns],
        characterised=list(characterised_any),
        protected=protected,
        row_metadata=row_metadata,
        stats=stats,
    )
    return bundle, warnings
