"""Single-process greedy versus exact diagnostic workflow."""

from __future__ import annotations

import csv
import json
import time
import uuid
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

try:  # pragma: no cover - exercised conditionally in tests
    from scipy import sparse
    from scipy.optimize import Bounds, LinearConstraint, milp

    SCIPY_MILP_AVAILABLE = True
except Exception:  # pragma: no cover - exercised conditionally in tests
    sparse = None
    Bounds = LinearConstraint = milp = None
    SCIPY_MILP_AVAILABLE = False

from .cf_resolution import CFResolutionManager
from .errors import DiagnosticConfigurationError
from .archive_reader import iter_source_entries, load_archive, merge_unit_registries
from .lcia import (
    collect_categories,
    ensure_category_factor_quality,
    resolve_lcia_archives,
    select_lcia_categories,
)
from .models import FlowInfo, ImpactCategory, UnitInfo
from .reducer import prepare_sparse_reduction_context
from .sparse_cover import column_nonzero_mask, retained_by_row, signed_tau_cover_sparse


ProgressCallback = Callable[[str, int, int], None]

DEFAULT_MAX_BINARY_VARIABLES = 250
DEFAULT_MAX_ACTIVE_ROWS = 1000
DEFAULT_SOLVER_TIME_LIMIT_SECONDS = 30.0
DEFAULT_MAX_TAU_VALUES = 5
DEFAULT_UNCHARACTERISED_POLICY = "keep"
DEFAULT_DIAGNOSTIC_ZIP_NAME = "greedy_exact_diagnostic.zip"
DEFAULT_METADATA_NAME = "greedy_exact_diagnostic_metadata.json"
DEFAULT_SUMMARY_CSV_NAME = "greedy_exact_diagnostic_summary.csv"
DEFAULT_DEBUG_CSV_NAME = "greedy_exact_diagnostic_exact_debug.csv"

SIGN_MODE_POSITIVE = "positive"
SIGN_MODE_NEGATIVE = "negative"
SIGN_MODE_BOTH = "both"
SIGN_MODES = {SIGN_MODE_POSITIVE, SIGN_MODE_NEGATIVE, SIGN_MODE_BOTH}

_SUMMARY_FIELDNAMES = [
    "process_id",
    "process_name",
    "process_path",
    "tau",
    "sign_mode",
    "candidate_exchanges",
    "protected_exchanges",
    "active_positive_rows",
    "active_negative_rows",
    "greedy_selector_positive_count",
    "exact_selector_positive_count",
    "positive_masks_identical",
    "positive_jaccard",
    "positive_only_greedy",
    "positive_only_exact",
    "greedy_selector_negative_count",
    "exact_selector_negative_count",
    "negative_masks_identical",
    "negative_jaccard",
    "negative_only_greedy",
    "negative_only_exact",
    "greedy_selector_union_count",
    "exact_selector_union_count",
    "union_masks_identical",
    "union_jaccard",
    "union_only_greedy",
    "union_only_exact",
    "protected_retained_by_policy_count",
    "greedy_exported_elementary_count",
    "exact_exported_elementary_count",
    "gap_exchanges",
    "gap_percent",
    "greedy_min_coverage",
    "exact_min_coverage",
    "greedy_certificate_pass",
    "exact_certificate_pass",
    "exact_solver_optimal",
    "exact_result_valid",
    "exact_status",
    "positive_status",
    "negative_status",
    "union_status",
    "runtime_seconds",
    "greedy_clone_name",
    "exact_clone_name",
    "exact_clone_written",
]

_WARNING_TEXT = (
    "The exact result is a single-process diagnostic for the selected LCIA rows and requested tau values. "
    "It does not prove global optimality of the whole reduced database. The diagnostic ZIP contains "
    "cloned process variants for inspection in openLCA; it does not replace the main database-scale "
    "tau-reduction workflow."
)


@dataclass(frozen=True)
class ExactSolveOutcome:
    status: str
    message: str
    selected_mask: np.ndarray
    runtime_seconds: float
    active_rows: int
    candidate_columns: int
    binary_variables: int
    constraints: int
    solver_optimal: bool = False
    certificate_pass: bool = False
    result_valid: bool = False
    solver_min_coverage: float | None = None
    verifier_min_coverage: float | None = None

    @property
    def is_optimal(self) -> bool:
        return self.solver_optimal


@dataclass
class TauDiagnosticResult:
    process_id: str
    process_name: str
    process_path: str
    tau: float
    sign_mode: str
    candidate_exchanges: int
    protected_exchanges: int
    protected_retained_by_policy_count: int
    active_positive_rows: int
    active_negative_rows: int
    greedy_positive_count: int
    greedy_negative_count: int
    greedy_selector_union_count: int
    greedy_union_count: int
    exact_positive_count: int | None
    exact_negative_count: int | None
    exact_selector_union_count: int | None
    exact_union_count: int | None
    exact_solver_optimal: bool
    exact_result_valid: bool
    exact_status: str
    positive_status: str
    negative_status: str
    union_status: str
    greedy_union_mask: np.ndarray
    exact_union_mask: np.ndarray | None
    greedy_min_coverage: float = 1.0
    exact_min_coverage: float | None = None
    greedy_certificate_pass: bool = True
    exact_certificate_pass: bool = False
    exact_runtime_seconds: float = 0.0
    greedy_clone_name: str = ""
    exact_clone_name: str = ""
    exact_clone_written: bool = False
    exact_debug_rows: tuple[dict[str, Any], ...] = ()
    exact_positive_solver_min_coverage: float | None = None
    exact_positive_verifier_min_coverage: float | None = None
    exact_negative_solver_min_coverage: float | None = None
    exact_negative_verifier_min_coverage: float | None = None
    greedy_positive_mask: np.ndarray | None = None
    greedy_negative_mask: np.ndarray | None = None
    greedy_selector_mask: np.ndarray | None = None
    exact_positive_mask: np.ndarray | None = None
    exact_negative_mask: np.ndarray | None = None
    exact_selector_mask: np.ndarray | None = None

    @property
    def gap_exchanges(self) -> int | None:
        if self.exact_union_count is None:
            return None
        return self.greedy_union_count - self.exact_union_count

    @property
    def gap_percent(self) -> float | None:
        return _gap_percent(self.greedy_union_count, self.exact_union_count)


@dataclass(frozen=True)
class GreedyExactDiagnosticConfig:
    database: str
    methods: Optional[str]
    output_dir: str
    method_selection: str
    process_query: str
    tau_values: Sequence[float]
    sign_mode: str = SIGN_MODE_BOTH
    strict_units: bool = True
    tolerance: float = 1e-12
    allow_water_mass_volume_override: bool = False
    max_scenario_rows_per_process: int = 300000
    max_candidate_set_size: int = 1000
    max_binary_variables: int = DEFAULT_MAX_BINARY_VARIABLES
    max_active_rows: int = DEFAULT_MAX_ACTIVE_ROWS
    solver_time_limit_seconds: float = DEFAULT_SOLVER_TIME_LIMIT_SECONDS


@dataclass
class GreedyExactDiagnosticResult:
    diagnostic_zip: str | None
    metadata_json: str
    summary_csv: str
    debug_csv: str | None
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class _PreparedProcessContext:
    process_id: str
    process_name: str
    process_path: str
    process_data: Dict[str, Any]
    flow_lookup: Dict[str, FlowInfo]
    unit_registry: Dict[str, UnitInfo]
    categories: Sequence[ImpactCategory]
    lcia_method_source: str
    internal_lcia_methods_ignored: bool
    empty_selected_categories: Sequence[Any]
    candidate_indices: Sequence[int]
    exchange_keys: Sequence[str]
    characterised_flags: Sequence[bool]
    protected_mask: np.ndarray
    protected_indices: Sequence[int]
    positive_model: Any
    negative_model: Any
    methods_descriptor: str

    @property
    def candidate_exchange_count(self) -> int:
        return len(self.candidate_indices)

    @property
    def protected_exchange_count(self) -> int:
        return len(self.protected_indices)


@dataclass(frozen=True)
class _LoadedDiagnosticInputs:
    database_archive: Any
    methods_archive: Any
    categories: Sequence[ImpactCategory]
    selected_categories: Sequence[ImpactCategory]
    unit_registry: Dict[str, UnitInfo]
    lcia_method_source: str
    internal_lcia_methods_ignored: bool
    empty_selected_categories: Sequence[Any]


def _emit_progress(
    progress_callback: Optional[ProgressCallback],
    message: str,
    current: int,
    total: int,
) -> None:
    if progress_callback is not None:
        progress_callback(message, current, total)


def _normalise_tau_values(values: Sequence[float]) -> List[float]:
    if not values:
        raise DiagnosticConfigurationError("Provide at least one tau value.")
    result: List[float] = []
    seen: set[str] = set()
    for value in sorted(float(item) for item in values):
        if value <= 0 or value > 1:
            raise DiagnosticConfigurationError("Tau values must be in (0, 1].")
        key = format(value, ".15g")
        if key in seen:
            raise DiagnosticConfigurationError("Tau values must be unique.")
        seen.add(key)
        result.append(value)
    if len(result) > DEFAULT_MAX_TAU_VALUES:
        raise DiagnosticConfigurationError(f"Provide at most {DEFAULT_MAX_TAU_VALUES} tau values.")
    return result


def _normalise_sign_mode(value: str) -> str:
    mode = str(value or "").strip().lower()
    if mode not in SIGN_MODES:
        raise DiagnosticConfigurationError("Sign mode must be one of: positive, negative, both.")
    return mode


def _serialise_json(data: Dict[str, Any]) -> bytes:
    return json.dumps(data, indent=2, ensure_ascii=True, sort_keys=False).encode("utf-8")


def _coverage_ratios(retained: np.ndarray, full: np.ndarray, active: np.ndarray) -> np.ndarray:
    coverage = np.ones_like(full, dtype=float)
    if active.any():
        coverage[active] = retained[active] / full[active]
    return coverage


def _coverage_requirement_satisfied(
    retained: np.ndarray,
    full: np.ndarray,
    active: np.ndarray,
    tau: float,
    tol: float,
) -> bool:
    if not active.any():
        return True
    coverage = _coverage_ratios(retained, full, active)
    return bool(np.all(coverage[active] + tol >= tau))


def _minimum_coverage(retained: np.ndarray, full: np.ndarray, active: np.ndarray) -> float:
    if not active.any():
        return 1.0
    coverage = _coverage_ratios(retained, full, active)
    return float(coverage[active].min())


def _dense_active_matrix(model: Any) -> np.ndarray:
    active_rows = np.flatnonzero(model.active)
    matrix = np.zeros((active_rows.size, model.n_cols), dtype=float)
    if active_rows.size == 0 or model.n_cols == 0:
        return matrix

    row_map = np.full(model.n_rows, -1, dtype=int)
    row_map[active_rows] = np.arange(active_rows.size)

    for block in model.exact_blocks:
        block_active = active_rows[(active_rows >= block.row_start) & (active_rows < block.row_stop)]
        if block_active.size == 0 or block.columns.size == 0:
            continue
        matrix[row_map[block_active][:, None], block.columns] += block.values[None, :]

    if model.scenario_row_indices.size:
        keep = model.active[model.scenario_row_indices]
        if np.any(keep):
            rows = row_map[model.scenario_row_indices[keep]]
            cols = model.scenario_col_indices[keep]
            values = model.scenario_values[keep]
            np.add.at(matrix, (rows, cols), values)
    return matrix


def _active_matrix_from_verifier(model: Any) -> np.ndarray:
    active_rows = int(model.active.sum())
    matrix = np.zeros((active_rows, model.n_cols), dtype=float)
    if active_rows == 0 or model.n_cols == 0:
        return matrix
    for column_index in range(model.n_cols):
        basis = np.zeros(model.n_cols, dtype=bool)
        basis[column_index] = True
        matrix[:, column_index] = retained_by_row(model, basis)[model.active]
    return matrix


def _normalised_active_matrix(model: Any, *, tolerance: float) -> tuple[np.ndarray, np.ndarray]:
    active_matrix = _active_matrix_from_verifier(model)
    full_active = np.asarray(model.full[model.active], dtype=float)
    if np.any(full_active <= tolerance):
        raise DiagnosticConfigurationError(
            "Active coverage rows must have strictly positive totals before exact normalisation."
        )
    return active_matrix / full_active[:, None], full_active


def _mask_array(mask: np.ndarray | Sequence[bool], n_cols: int, *, label: str) -> np.ndarray:
    array = np.asarray(mask, dtype=bool)
    if array.shape != (n_cols,):
        raise DiagnosticConfigurationError(
            f"{label} mask length mismatch: expected {n_cols}, got {array.shape}."
        )
    return array


def _matrix_equivalence_message(
    model: Any,
    *,
    active_matrix: np.ndarray,
    protected_mask: np.ndarray,
    tolerance: float,
    extra_masks: Sequence[np.ndarray] = (),
) -> str | None:
    if active_matrix.shape != (int(model.active.sum()), model.n_cols):
        return (
            "active matrix shape mismatch: "
            f"expected {(int(model.active.sum()), model.n_cols)}, got {active_matrix.shape}"
        )
    if active_matrix.size == 0:
        return None

    named_masks: list[tuple[str, np.ndarray]] = [
        ("empty", np.zeros(model.n_cols, dtype=bool)),
        ("protected_only", protected_mask.copy()),
        ("all_true", np.ones(model.n_cols, dtype=bool)),
    ]
    for index, mask in enumerate(extra_masks):
        named_masks.append((f"extra_{index}", _mask_array(mask, model.n_cols, label=f"extra_{index}")))
    rng = np.random.default_rng(0)
    for index in range(10):
        named_masks.append((f"random_{index}", rng.random(model.n_cols) >= 0.5))

    atol = max(float(tolerance), 1e-9)
    for label, mask in named_masks:
        lhs = active_matrix @ mask.astype(float)
        rhs = retained_by_row(model, mask)[model.active]
        if not np.allclose(lhs, rhs, atol=atol, rtol=0.0):
            diff = np.abs(lhs - rhs)
            row_index = int(np.argmax(diff))
            return (
                f"matrix/verifier mismatch for {label}: "
                f"active_row={row_index}; matrix_value={lhs[row_index]:.15g}; "
                f"verifier_value={rhs[row_index]:.15g}"
            )
    return None


def _decode_milp_solution(
    values: Any,
    *,
    n_cols: int,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    protected_mask: np.ndarray,
    tolerance: float,
) -> tuple[np.ndarray | None, str | None]:
    if values is None:
        return None, "solver returned no solution vector"
    solution = np.asarray(values, dtype=float)
    if solution.shape != (n_cols,):
        return None, f"solver returned shape {solution.shape}, expected {(n_cols,)}"
    if not np.all(np.isfinite(solution)):
        return None, "solver returned non-finite values"

    decode_tol = max(float(tolerance), 1e-7)
    if np.any(solution < lower_bounds - decode_tol) or np.any(solution > upper_bounds + decode_tol):
        return None, "solver solution violates variable bounds"
    rounded = np.rint(solution)
    if np.any(np.abs(solution - rounded) > decode_tol):
        return None, "solver solution is not numerically close to binary values"
    rounded = rounded.astype(int)
    if np.any(rounded < lower_bounds.astype(int)) or np.any(rounded > upper_bounds.astype(int)):
        return None, "rounded solver solution violates fixed variable bounds"
    selected = rounded.astype(bool)
    if np.any(~selected[protected_mask]):
        return None, "protected exchanges were not fixed to selected in the decoded solution"
    return selected, None


def _outcome_text(outcome: ExactSolveOutcome) -> str:
    if outcome.message:
        return f"{outcome.status}: {outcome.message}"
    return outcome.status


def _status_text(outcome: ExactSolveOutcome) -> str:
    if outcome.status == "optimal":
        return "optimal"
    if outcome.message:
        return f"{outcome.status}: {outcome.message}"
    return outcome.status


def _skipped_outcome(model: Any, message: str) -> ExactSolveOutcome:
    return ExactSolveOutcome(
        status="skipped",
        message=message,
        selected_mask=np.zeros(model.n_cols, dtype=bool),
        runtime_seconds=0.0,
        active_rows=int(model.active.sum()),
        candidate_columns=model.n_cols,
        binary_variables=0,
        constraints=int(model.active.sum()),
        solver_optimal=False,
        certificate_pass=False,
        result_valid=False,
    )


def _solve_exact_cover(
    model: Any,
    *,
    tau: float,
    tolerance: float,
    max_binary_variables: int,
    max_active_rows: int,
    solver_time_limit_seconds: float,
    equivalence_masks: Sequence[np.ndarray] = (),
) -> ExactSolveOutcome:
    active_rows = int(model.active.sum())
    candidate_columns = model.n_cols
    constraints = active_rows
    selected = np.zeros(model.n_cols, dtype=bool)
    runtime_seconds = 0.0
    raw_active_matrix = _active_matrix_from_verifier(model)
    matrix_message = _matrix_equivalence_message(
        model,
        active_matrix=raw_active_matrix,
        protected_mask=np.zeros(model.n_cols, dtype=bool),
        tolerance=tolerance,
        extra_masks=equivalence_masks,
    )
    if matrix_message is not None:
        return ExactSolveOutcome(
            status="internal_matrix_mismatch",
            message=matrix_message,
            selected_mask=selected,
            runtime_seconds=runtime_seconds,
            active_rows=active_rows,
            candidate_columns=candidate_columns,
            binary_variables=0,
            constraints=constraints,
            solver_optimal=False,
            certificate_pass=False,
            result_valid=False,
            solver_min_coverage=None,
            verifier_min_coverage=None,
        )

    if active_rows == 0:
        return ExactSolveOutcome(
            status="optimal",
            message="no active rows",
            selected_mask=selected,
            runtime_seconds=runtime_seconds,
            active_rows=active_rows,
            candidate_columns=candidate_columns,
            binary_variables=0,
            constraints=constraints,
            solver_optimal=True,
            certificate_pass=True,
            result_valid=True,
            solver_min_coverage=1.0,
            verifier_min_coverage=1.0,
        )
    if candidate_columns == 0:
        return ExactSolveOutcome(
            status="infeasible",
            message="active rows exist but the process has no candidate elementary exchanges",
            selected_mask=selected,
            runtime_seconds=runtime_seconds,
            active_rows=active_rows,
            candidate_columns=candidate_columns,
            binary_variables=0,
            constraints=constraints,
            solver_optimal=False,
            certificate_pass=False,
            result_valid=False,
            solver_min_coverage=None,
            verifier_min_coverage=None,
        )

    nonzero_columns = column_nonzero_mask(model)
    lower_bounds = np.zeros(candidate_columns, dtype=float)
    upper_bounds = np.where(nonzero_columns, 1.0, 0.0)
    objective = np.ones(candidate_columns, dtype=float)

    binary_variables = int(np.count_nonzero((lower_bounds == 0.0) & (upper_bounds == 1.0)))
    if binary_variables == 0:
        return ExactSolveOutcome(
            status="infeasible",
            message="no optimisable columns remain after removing zero-contribution exchanges",
            selected_mask=selected,
            runtime_seconds=runtime_seconds,
            active_rows=active_rows,
            candidate_columns=candidate_columns,
            binary_variables=binary_variables,
            constraints=constraints,
            solver_optimal=False,
            certificate_pass=False,
            result_valid=False,
            solver_min_coverage=None,
            verifier_min_coverage=None,
        )
    if active_rows > max_active_rows:
        return ExactSolveOutcome(
            status="limit_exceeded_active_rows",
            message=f"active_rows={active_rows} exceeds limit={max_active_rows}",
            selected_mask=selected,
            runtime_seconds=runtime_seconds,
            active_rows=active_rows,
            candidate_columns=candidate_columns,
            binary_variables=binary_variables,
            constraints=constraints,
            solver_optimal=False,
            certificate_pass=False,
            result_valid=False,
            solver_min_coverage=None,
            verifier_min_coverage=None,
        )
    if binary_variables > max_binary_variables:
        return ExactSolveOutcome(
            status="limit_exceeded_binary_variables",
            message=f"binary_variables={binary_variables} exceeds limit={max_binary_variables}",
            selected_mask=selected,
            runtime_seconds=runtime_seconds,
            active_rows=active_rows,
            candidate_columns=candidate_columns,
            binary_variables=binary_variables,
            constraints=constraints,
            solver_optimal=False,
            certificate_pass=False,
            result_valid=False,
            solver_min_coverage=None,
            verifier_min_coverage=None,
        )
    if not SCIPY_MILP_AVAILABLE:
        return ExactSolveOutcome(
            status="solver_unavailable",
            message="scipy.optimize.milp is unavailable",
            selected_mask=selected,
            runtime_seconds=runtime_seconds,
            active_rows=active_rows,
            candidate_columns=candidate_columns,
            binary_variables=binary_variables,
            constraints=constraints,
            solver_optimal=False,
            certificate_pass=False,
            result_valid=False,
            solver_min_coverage=None,
            verifier_min_coverage=None,
        )

    active_matrix, full_active = _normalised_active_matrix(model, tolerance=tolerance)
    lower_constraints = np.full(active_rows, float(tau), dtype=float)
    upper_constraints = np.full(active_rows, np.inf, dtype=float)

    started_at = time.monotonic()
    result = milp(
        c=objective,
        constraints=LinearConstraint(sparse.csr_matrix(active_matrix), lb=lower_constraints, ub=upper_constraints),
        integrality=np.ones(candidate_columns, dtype=int),
        bounds=Bounds(lower_bounds, upper_bounds),
        options={"time_limit": float(solver_time_limit_seconds)},
    )
    runtime_seconds = time.monotonic() - started_at

    status = int(getattr(result, "status", -1))
    message = str(getattr(result, "message", "") or "").strip()
    if status == 0 and getattr(result, "x", None) is not None:
        decoded_mask, decode_message = _decode_milp_solution(
            result.x,
            n_cols=candidate_columns,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            protected_mask=np.zeros(candidate_columns, dtype=bool),
            tolerance=tolerance,
        )
        if decoded_mask is None:
            return ExactSolveOutcome(
                status="invalid_exact_decode",
                message=decode_message or "failed to decode exact MILP solution",
                selected_mask=selected,
                runtime_seconds=runtime_seconds,
                active_rows=active_rows,
                candidate_columns=candidate_columns,
                binary_variables=binary_variables,
                constraints=constraints,
                solver_optimal=True,
                certificate_pass=False,
                result_valid=False,
                solver_min_coverage=None,
                verifier_min_coverage=None,
            )
        selected = decoded_mask
        solver_coverage = active_matrix @ selected.astype(float)
        solver_min_coverage = float(solver_coverage.min()) if solver_coverage.size else 1.0
        certificate_pass = _coverage_ok_for_sign(
            model,
            selected_mask=selected,
            tau=tau,
            tolerance=tolerance,
        )
        verifier_coverage = retained_by_row(model, selected)[model.active] / full_active
        verifier_min_coverage = float(verifier_coverage.min()) if verifier_coverage.size else 1.0
        if not certificate_pass:
            return ExactSolveOutcome(
                status="invalid_exact_certificate",
                message=(
                    f"solver returned optimal but coverage verification failed; "
                    f"solver_min_coverage={solver_min_coverage:.15g}; "
                    f"verifier_min_coverage={verifier_min_coverage:.15g}; tau={tau:.15g}"
                ),
                selected_mask=selected,
                runtime_seconds=runtime_seconds,
                active_rows=active_rows,
                candidate_columns=candidate_columns,
                binary_variables=binary_variables,
                constraints=constraints,
                solver_optimal=True,
                certificate_pass=False,
                result_valid=False,
                solver_min_coverage=solver_min_coverage,
                verifier_min_coverage=verifier_min_coverage,
            )
        return ExactSolveOutcome(
            status="optimal",
            message=message,
            selected_mask=selected,
            runtime_seconds=runtime_seconds,
            active_rows=active_rows,
            candidate_columns=candidate_columns,
            binary_variables=binary_variables,
            constraints=constraints,
            solver_optimal=True,
            certificate_pass=True,
            result_valid=True,
            solver_min_coverage=solver_min_coverage,
            verifier_min_coverage=verifier_min_coverage,
        )

    if status == 1:
        status_text = "iteration_limit"
    elif status == 2:
        status_text = "infeasible"
    elif status == 3:
        status_text = "unbounded"
    elif status == 4:
        status_text = "solver_error"
    else:
        status_text = "not_optimal"
    if "time" in message.lower():
        status_text = "time_limit"
    return ExactSolveOutcome(
        status=status_text,
        message=message,
        selected_mask=selected,
        runtime_seconds=runtime_seconds,
        active_rows=active_rows,
        candidate_columns=candidate_columns,
        binary_variables=binary_variables,
        constraints=constraints,
        solver_optimal=False,
        certificate_pass=False,
        result_valid=False,
        solver_min_coverage=None,
        verifier_min_coverage=None,
    )


def _selected_process_methods_descriptor(categories: Sequence[ImpactCategory], method_selection: str) -> str:
    labels = []
    for category in categories[:10]:
        method_name = category.method_name or category.method_id or "-"
        labels.append(f"{method_name} / {category.name}")
    if len(categories) > 10:
        labels.append(f"... ({len(categories)} categories total)")
    joined = "; ".join(labels) if labels else "-"
    return f"selection={method_selection}; categories={len(categories)}; {joined}"


def _resolve_process_entry(database_archive: Any, query: str) -> Any:
    token = str(query or "").strip()
    if not token:
        raise DiagnosticConfigurationError("Enter a process name or UUID.")

    by_id = database_archive.processes.get(token)
    if by_id is not None:
        return by_id

    exact_name_matches = [entry for entry in database_archive.processes.values() if (entry.name or "").strip() == token]
    if len(exact_name_matches) == 1:
        return exact_name_matches[0]
    if len(exact_name_matches) > 1:
        raise DiagnosticConfigurationError(
            f"Process name '{token}' is ambiguous. Provide the process UUID instead."
        )

    lowered = token.casefold()
    casefold_name_matches = [entry for entry in database_archive.processes.values() if (entry.name or "").strip().casefold() == lowered]
    if len(casefold_name_matches) == 1:
        return casefold_name_matches[0]
    if len(casefold_name_matches) > 1:
        raise DiagnosticConfigurationError(
            f"Process name '{token}' is ambiguous after case-insensitive matching. Provide the process UUID instead."
        )

    raise DiagnosticConfigurationError(
        f"Process '{token}' was not found. Enter the exact process name or UUID."
    )


def _resolve_process_entries(database_archive: Any, query: str) -> list[Any]:
    token = str(query or "").strip()
    if not token:
        return sorted(
            database_archive.processes.values(),
            key=lambda entry: ((entry.name or "").casefold(), str(entry.object_id)),
        )
    return [_resolve_process_entry(database_archive, token)]


def _validate_candidate_mapping(exchanges: Sequence[Dict[str, Any]], candidate_indices: Sequence[int]) -> None:
    seen: set[int] = set()
    for index in candidate_indices:
        candidate_index = int(index)
        if candidate_index < 0 or candidate_index >= len(exchanges):
            raise DiagnosticConfigurationError(
                f"Candidate exchange index {candidate_index} is outside the process exchange list."
            )
        if candidate_index in seen:
            raise DiagnosticConfigurationError(
                f"Candidate exchange index {candidate_index} is duplicated in the contribution model."
            )
        seen.add(candidate_index)


def _load_diagnostic_inputs(config: GreedyExactDiagnosticConfig) -> _LoadedDiagnosticInputs:
    database_archive = load_archive(config.database, require_processes=True, require_flows=True, conversion_scope="inspect")
    methods_archive = (
        load_archive(config.methods, require_processes=False, require_flows=False, conversion_scope="inspect")
        if config.methods
        else None
    )
    archives, lcia_method_source, internal_lcia_methods_ignored = resolve_lcia_archives(database_archive, methods_archive)
    categories = collect_categories(archives, diagnostic_file=DEFAULT_METADATA_NAME)
    selected_categories = select_lcia_categories(categories, config.method_selection)
    empty_selected_categories = ensure_category_factor_quality(selected_categories)
    return _LoadedDiagnosticInputs(
        database_archive=database_archive,
        methods_archive=methods_archive,
        categories=categories,
        selected_categories=selected_categories,
        unit_registry=merge_unit_registries(
            getattr(database_archive, "units", {}),
            getattr(methods_archive, "units", {}) if methods_archive is not None else {},
        ),
        lcia_method_source=lcia_method_source,
        internal_lcia_methods_ignored=internal_lcia_methods_ignored,
        empty_selected_categories=empty_selected_categories,
    )


def _prepare_context_for_entry(
    config: GreedyExactDiagnosticConfig,
    loaded: _LoadedDiagnosticInputs,
    process_entry: Any,
) -> _PreparedProcessContext:
    process_data = deepcopy(process_entry.data)
    reduction_context = prepare_sparse_reduction_context(
        process_data,
        loaded.database_archive.flows,
        loaded.selected_categories,
        uncharacterised_policy=DEFAULT_UNCHARACTERISED_POLICY,
        strict_units=config.strict_units,
        tol=config.tolerance,
        allow_water_mass_volume_override=config.allow_water_mass_volume_override,
        unit_registry=loaded.unit_registry,
        warning_records=None,
        ambiguity_records=None,
        diagnostic_file=DEFAULT_METADATA_NAME,
        resolution_manager=CFResolutionManager(mode="cli"),
        max_scenario_rows_per_process=config.max_scenario_rows_per_process,
        max_candidate_set_size=config.max_candidate_set_size,
        include_row_metadata=True,
    )
    sparse_details = reduction_context.sparse_details
    exchanges = reduction_context.exchanges
    _validate_candidate_mapping(exchanges, reduction_context.candidate_indices)
    protected = reduction_context.protected_mask.copy()
    protected_indices = list(np.flatnonzero(protected))

    return _PreparedProcessContext(
        process_id=str(process_entry.object_id),
        process_name=str(process_entry.name or process_entry.object_id),
        process_path=process_entry.path,
        process_data=process_data,
        flow_lookup=loaded.database_archive.flows,
        unit_registry=loaded.unit_registry,
        categories=loaded.selected_categories,
        lcia_method_source=loaded.lcia_method_source,
        internal_lcia_methods_ignored=loaded.internal_lcia_methods_ignored,
        empty_selected_categories=loaded.empty_selected_categories,
        candidate_indices=tuple(reduction_context.candidate_indices),
        exchange_keys=tuple(reduction_context.exchange_keys),
        characterised_flags=tuple(reduction_context.characterised_flags),
        protected_mask=protected,
        protected_indices=tuple(protected_indices),
        positive_model=reduction_context.positive_model,
        negative_model=reduction_context.negative_model,
        methods_descriptor=_selected_process_methods_descriptor(loaded.selected_categories, config.method_selection),
    )


def _prepare_context(config: GreedyExactDiagnosticConfig) -> _PreparedProcessContext:
    loaded = _load_diagnostic_inputs(config)
    process_entry = _resolve_process_entry(loaded.database_archive, config.process_query)
    return _prepare_context_for_entry(config, loaded, process_entry)


def _coverage_ok_for_sign(
    model: Any,
    *,
    selected_mask: np.ndarray,
    tau: float,
    tolerance: float,
) -> bool:
    retained = retained_by_row(model, selected_mask)
    return _coverage_requirement_satisfied(retained, model.full, model.active, tau, tolerance)


def _minimum_coverage_for_sign(model: Any, selected_mask: np.ndarray) -> float:
    retained = retained_by_row(model, selected_mask)
    return _minimum_coverage(retained, model.full, model.active)


def _gap_percent(greedy_retained: int, exact_retained: int | None) -> float | None:
    if exact_retained is None or exact_retained <= 0:
        return None
    return 100.0 * (greedy_retained - exact_retained) / exact_retained


def _mask_similarity(greedy_mask: np.ndarray | None, exact_mask: np.ndarray | None) -> dict[str, Any]:
    if greedy_mask is None or exact_mask is None:
        return {
            "identical": None,
            "jaccard": None,
            "only_greedy": None,
            "only_exact": None,
        }
    greedy = np.asarray(greedy_mask, dtype=bool)
    exact = np.asarray(exact_mask, dtype=bool)
    identical = bool(np.array_equal(greedy, exact))
    union = int(np.count_nonzero(greedy | exact))
    intersection = int(np.count_nonzero(greedy & exact))
    jaccard = 1.0 if union == 0 else intersection / union
    return {
        "identical": identical,
        "jaccard": float(jaccard),
        "only_greedy": int(np.count_nonzero(greedy & ~exact)),
        "only_exact": int(np.count_nonzero(exact & ~greedy)),
    }


def _selected_exchange_ids(exchange_keys: Sequence[str], mask: np.ndarray | None) -> list[str] | None:
    if mask is None:
        return None
    selected = np.asarray(mask, dtype=bool)
    return [str(exchange_keys[index]) for index, keep in enumerate(selected) if keep]


def _invalidate_exact_outcome(outcome: ExactSolveOutcome, *, status: str, message: str) -> ExactSolveOutcome:
    return replace(
        outcome,
        status=status,
        message=message,
        certificate_pass=False,
        result_valid=False,
    )


def _union_status(positive: ExactSolveOutcome, negative: ExactSolveOutcome, sign_mode: str) -> str:
    if sign_mode == SIGN_MODE_POSITIVE:
        return _outcome_text(positive)
    if sign_mode == SIGN_MODE_NEGATIVE:
        return _outcome_text(negative)
    if positive.result_valid and negative.result_valid:
        return "valid_exact_result"
    return f"positive={_outcome_text(positive)} | negative={_outcome_text(negative)}"


def _evaluate_tau(
    context: _PreparedProcessContext,
    *,
    tau: float,
    sign_mode: str,
    config: GreedyExactDiagnosticConfig,
    progress_callback: Optional[ProgressCallback],
    current: int,
    total: int,
) -> TauDiagnosticResult:
    candidate_exchanges = context.candidate_exchange_count
    protected = _mask_array(context.protected_mask, candidate_exchanges, label="protected")
    exchange_keys = context.exchange_keys
    tol = config.tolerance

    _emit_progress(
        progress_callback,
        (
            f"Tau {format(tau, '.15g')}: candidate exchanges={candidate_exchanges}, "
            f"protected exchanges={context.protected_exchange_count}, "
            f"positive rows={int(context.positive_model.active.sum())}, "
            f"negative rows={int(context.negative_model.active.sum())}"
        ),
        current,
        total,
    )

    greedy_cover = signed_tau_cover_sparse(
        context.positive_model,
        context.negative_model,
        tau=tau,
        exchange_keys=exchange_keys,
        tol=tol,
    )
    greedy_positive = _mask_array(greedy_cover["selected_pos"], candidate_exchanges, label="greedy_positive")
    greedy_negative = _mask_array(greedy_cover["selected_neg"], candidate_exchanges, label="greedy_negative")
    greedy_selected = _mask_array(greedy_cover["selected"], candidate_exchanges, label="greedy_selected")

    positive_outcome = (
        _solve_exact_cover(
            context.positive_model,
            tau=tau,
            tolerance=tol,
            max_binary_variables=config.max_binary_variables,
            max_active_rows=config.max_active_rows,
            solver_time_limit_seconds=config.solver_time_limit_seconds,
            equivalence_masks=(greedy_positive,),
        )
        if sign_mode in {SIGN_MODE_POSITIVE, SIGN_MODE_BOTH}
        else _skipped_outcome(context.positive_model, "sign mode excludes positive exact solve")
    )
    negative_outcome = (
        _solve_exact_cover(
            context.negative_model,
            tau=tau,
            tolerance=tol,
            max_binary_variables=config.max_binary_variables,
            max_active_rows=config.max_active_rows,
            solver_time_limit_seconds=config.solver_time_limit_seconds,
            equivalence_masks=(greedy_negative,),
        )
        if sign_mode in {SIGN_MODE_NEGATIVE, SIGN_MODE_BOTH}
        else _skipped_outcome(context.negative_model, "sign mode excludes negative exact solve")
    )
    _emit_progress(
        progress_callback,
        (
            f"Tau {format(tau, '.15g')}: positive exact model columns={candidate_exchanges}, "
            f"active_rows={positive_outcome.active_rows}, binary_variables={positive_outcome.binary_variables}, "
            f"constraints={positive_outcome.constraints}, status={_outcome_text(positive_outcome)}; "
            f"negative exact model columns={candidate_exchanges}, active_rows={negative_outcome.active_rows}, "
            f"binary_variables={negative_outcome.binary_variables}, constraints={negative_outcome.constraints}, "
            f"status={_outcome_text(negative_outcome)}"
        ),
        current,
        total,
    )

    greedy_export_mask = greedy_selected | protected

    positive_greedy_cover_ok = _coverage_ok_for_sign(
        context.positive_model,
        selected_mask=greedy_export_mask,
        tau=tau,
        tolerance=tol,
    )
    negative_greedy_cover_ok = _coverage_ok_for_sign(
        context.negative_model,
        selected_mask=greedy_export_mask,
        tau=tau,
        tolerance=tol,
    )

    if positive_outcome.result_valid and positive_greedy_cover_ok:
        exact_positive_count = int(positive_outcome.selected_mask.sum())
        if exact_positive_count > int(greedy_positive.sum()):
            positive_outcome = _invalidate_exact_outcome(
                positive_outcome,
                status="invalid_exact_dominance",
                message=(
                    "positive exact selector retained more exchanges than the greedy positive selector "
                    f"({exact_positive_count} > {int(greedy_positive.sum())})"
                ),
            )
    if negative_outcome.result_valid and negative_greedy_cover_ok:
        exact_negative_count = int(negative_outcome.selected_mask.sum())
        if exact_negative_count > int(greedy_negative.sum()):
            negative_outcome = _invalidate_exact_outcome(
                negative_outcome,
                status="invalid_exact_dominance",
                message=(
                    "negative exact selector retained more exchanges than the greedy negative selector "
                    f"({exact_negative_count} > {int(greedy_negative.sum())})"
                ),
            )

    exact_positive_selector = positive_outcome.selected_mask if positive_outcome.result_valid else None
    exact_negative_selector = negative_outcome.selected_mask if negative_outcome.result_valid else None

    exact_selector_union = None
    exact_export_mask = None
    exact_solver_optimal = False
    exact_result_valid = False
    if sign_mode == SIGN_MODE_POSITIVE:
        exact_solver_optimal = positive_outcome.solver_optimal
        exact_result_valid = positive_outcome.result_valid
        exact_selector_union = exact_positive_selector
    elif sign_mode == SIGN_MODE_NEGATIVE:
        exact_solver_optimal = negative_outcome.solver_optimal
        exact_result_valid = negative_outcome.result_valid
        exact_selector_union = exact_negative_selector
    else:
        exact_solver_optimal = positive_outcome.solver_optimal and negative_outcome.solver_optimal
        exact_result_valid = positive_outcome.result_valid and negative_outcome.result_valid
        if exact_result_valid:
            exact_selector_union = exact_positive_selector | exact_negative_selector
    if exact_selector_union is not None:
        exact_export_mask = exact_selector_union | protected

    if sign_mode == SIGN_MODE_POSITIVE:
        greedy_certificate_pass = positive_greedy_cover_ok
        exact_certificate_pass = positive_outcome.certificate_pass and positive_outcome.result_valid
        greedy_min_coverage = _minimum_coverage_for_sign(context.positive_model, greedy_export_mask)
        exact_min_coverage = (
            _minimum_coverage_for_sign(context.positive_model, exact_export_mask)
            if exact_export_mask is not None
            else None
        )
    elif sign_mode == SIGN_MODE_NEGATIVE:
        greedy_certificate_pass = negative_greedy_cover_ok
        exact_certificate_pass = negative_outcome.certificate_pass and negative_outcome.result_valid
        greedy_min_coverage = _minimum_coverage_for_sign(context.negative_model, greedy_export_mask)
        exact_min_coverage = (
            _minimum_coverage_for_sign(context.negative_model, exact_export_mask)
            if exact_export_mask is not None
            else None
        )
    else:
        greedy_certificate_pass = positive_greedy_cover_ok and negative_greedy_cover_ok
        exact_certificate_pass = (
            positive_outcome.certificate_pass
            and negative_outcome.certificate_pass
            and positive_outcome.result_valid
            and negative_outcome.result_valid
        )
        greedy_min_coverage = min(
            _minimum_coverage_for_sign(context.positive_model, greedy_export_mask),
            _minimum_coverage_for_sign(context.negative_model, greedy_export_mask),
        )
        exact_min_coverage = (
            min(
                _minimum_coverage_for_sign(context.positive_model, exact_export_mask),
                _minimum_coverage_for_sign(context.negative_model, exact_export_mask),
            )
            if exact_export_mask is not None
            else None
        )
    exact_positive_count = int(positive_outcome.selected_mask.sum()) if positive_outcome.result_valid else None
    exact_negative_count = int(negative_outcome.selected_mask.sum()) if negative_outcome.result_valid else None
    greedy_selector_union_count = int(greedy_selected.sum())
    exact_selector_union_count = int(exact_selector_union.sum()) if exact_selector_union is not None else None
    exact_status = _union_status(positive_outcome, negative_outcome, sign_mode)
    exact_debug_rows: list[dict[str, Any]] = []
    if not positive_outcome.result_valid and sign_mode in {SIGN_MODE_POSITIVE, SIGN_MODE_BOTH}:
        positive_normalised, _ = _normalised_active_matrix(context.positive_model, tolerance=tol)
        for row in _debug_rows_for_mask(
            context.positive_model,
            sign_label="positive",
            tau=tau,
            selected_mask=positive_outcome.selected_mask,
            normalised_matrix=positive_normalised,
        ):
            row["tau"] = format(tau, ".15g")
            row["mask_type"] = "exact_invalid"
            exact_debug_rows.append(row)
    if not negative_outcome.result_valid and sign_mode in {SIGN_MODE_NEGATIVE, SIGN_MODE_BOTH}:
        negative_normalised, _ = _normalised_active_matrix(context.negative_model, tolerance=tol)
        for row in _debug_rows_for_mask(
            context.negative_model,
            sign_label="negative",
            tau=tau,
            selected_mask=negative_outcome.selected_mask,
            normalised_matrix=negative_normalised,
        ):
            row["tau"] = format(tau, ".15g")
            row["mask_type"] = "exact_invalid"
            exact_debug_rows.append(row)
    return TauDiagnosticResult(
        process_id=context.process_id,
        process_name=context.process_name,
        process_path=context.process_path,
        tau=tau,
        sign_mode=sign_mode,
        candidate_exchanges=candidate_exchanges,
        protected_exchanges=context.protected_exchange_count,
        protected_retained_by_policy_count=context.protected_exchange_count,
        active_positive_rows=int(context.positive_model.active.sum()),
        active_negative_rows=int(context.negative_model.active.sum()),
        greedy_positive_count=int(greedy_positive.sum()),
        greedy_negative_count=int(greedy_negative.sum()),
        greedy_selector_union_count=greedy_selector_union_count,
        greedy_union_count=int(greedy_export_mask.sum()),
        exact_positive_count=exact_positive_count,
        exact_negative_count=exact_negative_count,
        exact_selector_union_count=exact_selector_union_count,
        exact_union_count=int(exact_export_mask.sum()) if exact_export_mask is not None else None,
        exact_solver_optimal=exact_solver_optimal,
        exact_result_valid=exact_result_valid,
        exact_status=exact_status,
        exact_positive_solver_min_coverage=positive_outcome.solver_min_coverage,
        exact_positive_verifier_min_coverage=positive_outcome.verifier_min_coverage,
        exact_negative_solver_min_coverage=negative_outcome.solver_min_coverage,
        exact_negative_verifier_min_coverage=negative_outcome.verifier_min_coverage,
        positive_status=_outcome_text(positive_outcome),
        negative_status=_outcome_text(negative_outcome),
        union_status=exact_status,
        greedy_union_mask=greedy_export_mask,
        exact_union_mask=exact_export_mask,
        greedy_min_coverage=greedy_min_coverage,
        exact_min_coverage=exact_min_coverage,
        greedy_certificate_pass=greedy_certificate_pass,
        exact_certificate_pass=exact_certificate_pass,
        exact_runtime_seconds=positive_outcome.runtime_seconds + negative_outcome.runtime_seconds,
        exact_debug_rows=tuple(exact_debug_rows),
        greedy_positive_mask=greedy_positive.copy(),
        greedy_negative_mask=greedy_negative.copy(),
        greedy_selector_mask=greedy_selected.copy(),
        exact_positive_mask=None if exact_positive_selector is None else exact_positive_selector.copy(),
        exact_negative_mask=None if exact_negative_selector is None else exact_negative_selector.copy(),
        exact_selector_mask=None if exact_selector_union is None else exact_selector_union.copy(),
    )


def _tau_suffix(tau: float) -> str:
    return format(tau, ".15g").replace(".", "_")


def _unique_clone_name(base_name: str, existing_names: set[str], tau: float) -> str:
    if base_name not in existing_names:
        existing_names.add(base_name)
        return base_name
    candidate = f"{base_name}_tau_{_tau_suffix(tau)}"
    if candidate not in existing_names:
        existing_names.add(candidate)
        return candidate
    index = 2
    while f"{candidate}_{index}" in existing_names:
        index += 1
    final_name = f"{candidate}_{index}"
    existing_names.add(final_name)
    return final_name


def _default_clone_name(process_name: str, clone_kind: str, tau: float, *, multiple_tau: bool) -> str:
    if multiple_tau:
        return f"{process_name}_{clone_kind}_tau_{_tau_suffix(tau)}"
    return f"{process_name}_{clone_kind}"


def _category_descriptor(categories: Sequence[ImpactCategory]) -> str:
    lines = []
    for category in categories[:10]:
        method_name = category.method_name or category.method_id or "-"
        lines.append(f"- {method_name} / {category.name}")
    if len(categories) > 10:
        lines.append(f"- ... ({len(categories)} categories total)")
    return "\n".join(lines) if lines else "- none -"


def _clone_process_with_selection(
    process_data: Dict[str, Any],
    *,
    clone_name: str,
    clone_kind: str,
    selected_mask: np.ndarray,
    context: _PreparedProcessContext,
    tau: float,
    sign_mode: str,
    solver_type: str,
    created_at: str,
) -> Dict[str, Any]:
    _mask_array(selected_mask, len(context.candidate_indices), label=f"{clone_kind}_clone_selection")
    _validate_candidate_mapping(list(process_data.get("exchanges") or []), context.candidate_indices)
    clone = deepcopy(process_data)
    new_uuid = str(uuid.uuid4())
    for key in ("@id", "id", "refId", "uuid"):
        if key in clone:
            clone[key] = new_uuid
    clone["name"] = clone_name

    exchanges = list(clone.get("exchanges") or [])
    kept_exchanges: List[Dict[str, Any]] = []
    candidate_lookup = {int(candidate_index): local_index for local_index, candidate_index in enumerate(context.candidate_indices)}
    removed_elementary = 0
    kept_elementary = 0
    for exchange_index, exchange in enumerate(exchanges):
        local_index = candidate_lookup.get(exchange_index)
        if local_index is None:
            kept_exchanges.append(exchange)
            continue
        if bool(selected_mask[local_index]):
            kept_exchanges.append(exchange)
            kept_elementary += 1
        else:
            removed_elementary += 1
    clone["exchanges"] = kept_exchanges

    selected_label = _category_descriptor(context.categories)
    positive_coverage = _minimum_coverage_for_sign(context.positive_model, selected_mask)
    negative_coverage = _minimum_coverage_for_sign(context.negative_model, selected_mask)
    if sign_mode == SIGN_MODE_POSITIVE:
        minimum_coverage = positive_coverage
    elif sign_mode == SIGN_MODE_NEGATIVE:
        minimum_coverage = negative_coverage
    else:
        minimum_coverage = min(positive_coverage, negative_coverage)

    description_lines = [
        "Diagnostic clone created by lci_reduce.",
        f"Original process name: {context.process_name}",
        f"Original process UUID: {context.process_id}",
        f"Diagnostic clone type: {clone_kind}",
        f"Tau: {format(tau, '.15g')}",
        f"Selected LCIA methods/categories:\n{selected_label}",
        f"Solver type: {solver_type}",
        f"Sign mode: {sign_mode}",
        f"Retained elementary exchange count: {kept_elementary}",
        f"Removed elementary exchange count: {removed_elementary}",
        f"Minimum coverage achieved: {format(minimum_coverage, '.15g')}",
        f"Creation timestamp: {created_at}",
        "This process is a diagnostic clone.",
    ]
    existing_description = str(clone.get("description") or "").strip()
    if existing_description:
        description_lines.insert(0, existing_description)
    clone["description"] = "\n\n".join(description_lines)
    return clone


def _clone_process_path(original_path: str, clone_uuid: str) -> str:
    original = Path(original_path)
    return original.with_name(f"{clone_uuid}{original.suffix or '.json'}").as_posix()


def _write_diagnostic_zip(
    path: Path,
    *,
    database_source: str,
    selected_process_path: str,
    clones: Sequence[Dict[str, Any]],
) -> None:
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        for rel_path, raw_bytes in iter_source_entries(database_source):
            archive.writestr(rel_path, raw_bytes)
        for clone in clones:
            clone_id = str(clone.get("@id") or clone.get("id"))
            clone_path = _clone_process_path(selected_process_path, clone_id)
            archive.writestr(clone_path, _serialise_json(clone))


def _write_summary_csv(path: Path, tau_results: Sequence[TauDiagnosticResult]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_SUMMARY_FIELDNAMES)
        writer.writeheader()
        for result in tau_results:
            positive_similarity = _mask_similarity(result.greedy_positive_mask, result.exact_positive_mask)
            negative_similarity = _mask_similarity(result.greedy_negative_mask, result.exact_negative_mask)
            union_similarity = _mask_similarity(result.greedy_selector_mask, result.exact_selector_mask)
            writer.writerow(
                {
                    "process_id": result.process_id,
                    "process_name": result.process_name,
                    "process_path": result.process_path,
                    "tau": format(result.tau, ".15g"),
                    "sign_mode": result.sign_mode,
                    "candidate_exchanges": result.candidate_exchanges,
                    "protected_exchanges": result.protected_exchanges,
                    "active_positive_rows": result.active_positive_rows,
                    "active_negative_rows": result.active_negative_rows,
                    "greedy_selector_positive_count": result.greedy_positive_count,
                    "exact_selector_positive_count": "" if result.exact_positive_count is None else result.exact_positive_count,
                    "positive_masks_identical": "" if positive_similarity["identical"] is None else positive_similarity["identical"],
                    "positive_jaccard": "" if positive_similarity["jaccard"] is None else format(positive_similarity["jaccard"], ".6f"),
                    "positive_only_greedy": "" if positive_similarity["only_greedy"] is None else positive_similarity["only_greedy"],
                    "positive_only_exact": "" if positive_similarity["only_exact"] is None else positive_similarity["only_exact"],
                    "greedy_selector_negative_count": result.greedy_negative_count,
                    "exact_selector_negative_count": "" if result.exact_negative_count is None else result.exact_negative_count,
                    "negative_masks_identical": "" if negative_similarity["identical"] is None else negative_similarity["identical"],
                    "negative_jaccard": "" if negative_similarity["jaccard"] is None else format(negative_similarity["jaccard"], ".6f"),
                    "negative_only_greedy": "" if negative_similarity["only_greedy"] is None else negative_similarity["only_greedy"],
                    "negative_only_exact": "" if negative_similarity["only_exact"] is None else negative_similarity["only_exact"],
                    "greedy_selector_union_count": result.greedy_selector_union_count,
                    "exact_selector_union_count": "" if result.exact_selector_union_count is None else result.exact_selector_union_count,
                    "union_masks_identical": "" if union_similarity["identical"] is None else union_similarity["identical"],
                    "union_jaccard": "" if union_similarity["jaccard"] is None else format(union_similarity["jaccard"], ".6f"),
                    "union_only_greedy": "" if union_similarity["only_greedy"] is None else union_similarity["only_greedy"],
                    "union_only_exact": "" if union_similarity["only_exact"] is None else union_similarity["only_exact"],
                    "protected_retained_by_policy_count": result.protected_retained_by_policy_count,
                    "greedy_exported_elementary_count": result.greedy_union_count,
                    "exact_exported_elementary_count": "" if result.exact_union_count is None else result.exact_union_count,
                    "gap_exchanges": "" if result.gap_exchanges is None else result.gap_exchanges,
                    "gap_percent": "" if result.gap_percent is None else format(result.gap_percent, ".6f"),
                    "greedy_min_coverage": format(result.greedy_min_coverage, ".15g"),
                    "exact_min_coverage": "" if result.exact_min_coverage is None else format(result.exact_min_coverage, ".15g"),
                    "greedy_certificate_pass": result.greedy_certificate_pass,
                    "exact_certificate_pass": result.exact_certificate_pass,
                    "exact_solver_optimal": result.exact_solver_optimal,
                    "exact_result_valid": result.exact_result_valid,
                    "exact_status": result.exact_status,
                    "positive_status": result.positive_status,
                    "negative_status": result.negative_status,
                    "union_status": result.union_status,
                    "runtime_seconds": format(result.exact_runtime_seconds, ".6f"),
                    "greedy_clone_name": result.greedy_clone_name,
                    "exact_clone_name": result.exact_clone_name,
                    "exact_clone_written": result.exact_clone_written,
                }
            )


def _metadata_warning(message: str) -> Dict[str, str]:
    return {"severity": "warning", "message": message}


def _debug_rows_for_mask(
    model: Any,
    *,
    sign_label: str,
    tau: float,
    selected_mask: np.ndarray,
    normalised_matrix: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    retained = retained_by_row(model, selected_mask)
    rows: list[dict[str, Any]] = []
    active_indices = np.flatnonzero(model.active)
    selector = selected_mask.astype(float)
    solver_coverage = normalised_matrix @ selector if normalised_matrix is not None else None
    for local_active_index, row_index in enumerate(active_indices):
        full = float(model.full[row_index])
        demand = float(tau * full)
        retained_value = float(retained[row_index])
        verifier_coverage = retained_value / full if full > 0 else 1.0
        rows.append(
            {
                "sign": sign_label,
                "active_row_index": local_active_index,
                "model_row_index": int(row_index),
                "full": format(full, ".15g"),
                "demand": format(demand, ".15g"),
                "retained": format(retained_value, ".15g"),
                "solver_coverage": "" if solver_coverage is None else format(float(solver_coverage[local_active_index]), ".15g"),
                "verifier_coverage": format(verifier_coverage, ".15g"),
            }
        )
    return rows


def _write_debug_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fieldnames = [
        "tau",
        "mask_type",
        "sign",
        "active_row_index",
        "model_row_index",
        "full",
        "demand",
        "retained",
        "solver_coverage",
        "verifier_coverage",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _serialise_tau_result(context: _PreparedProcessContext, result: TauDiagnosticResult) -> dict[str, Any]:
    return {
        "process_id": context.process_id,
        "process_name": context.process_name,
        "process_path": context.process_path,
        "selector_comparison": {
            "positive": _mask_similarity(result.greedy_positive_mask, result.exact_positive_mask),
            "negative": _mask_similarity(result.greedy_negative_mask, result.exact_negative_mask),
            "union": _mask_similarity(result.greedy_selector_mask, result.exact_selector_mask),
        },
        "selected_exchange_ids": {
            "greedy_positive": _selected_exchange_ids(context.exchange_keys, result.greedy_positive_mask),
            "exact_positive": _selected_exchange_ids(context.exchange_keys, result.exact_positive_mask),
            "greedy_negative": _selected_exchange_ids(context.exchange_keys, result.greedy_negative_mask),
            "exact_negative": _selected_exchange_ids(context.exchange_keys, result.exact_negative_mask),
            "greedy_export": _selected_exchange_ids(context.exchange_keys, result.greedy_union_mask),
            "exact_export": _selected_exchange_ids(context.exchange_keys, result.exact_union_mask),
        },
        "tau": result.tau,
        "sign_mode": result.sign_mode,
        "candidate_exchanges": result.candidate_exchanges,
        "protected_exchanges": result.protected_exchanges,
        "active_positive_rows": result.active_positive_rows,
        "active_negative_rows": result.active_negative_rows,
        "greedy_positive_count": result.greedy_positive_count,
        "greedy_negative_count": result.greedy_negative_count,
        "greedy_selector_union_count": result.greedy_selector_union_count,
        "greedy_union_count": result.greedy_union_count,
        "exact_positive_count": result.exact_positive_count,
        "exact_negative_count": result.exact_negative_count,
        "exact_selector_union_count": result.exact_selector_union_count,
        "exact_union_count": result.exact_union_count,
        "protected_retained_by_policy_count": result.protected_retained_by_policy_count,
        "gap_exchanges": result.gap_exchanges,
        "gap_percent": result.gap_percent,
        "greedy_min_coverage": result.greedy_min_coverage,
        "exact_min_coverage": result.exact_min_coverage,
        "greedy_certificate_pass": result.greedy_certificate_pass,
        "exact_certificate_pass": result.exact_certificate_pass,
        "exact_solver_optimal": result.exact_solver_optimal,
        "exact_result_valid": result.exact_result_valid,
        "exact_status": result.exact_status,
        "exact_positive_solver_min_coverage": result.exact_positive_solver_min_coverage,
        "exact_positive_verifier_min_coverage": result.exact_positive_verifier_min_coverage,
        "exact_negative_solver_min_coverage": result.exact_negative_solver_min_coverage,
        "exact_negative_verifier_min_coverage": result.exact_negative_verifier_min_coverage,
        "positive_status": result.positive_status,
        "negative_status": result.negative_status,
        "union_status": result.union_status,
        "greedy_clone_name": result.greedy_clone_name,
        "exact_clone_name": result.exact_clone_name,
        "exact_clone_written": result.exact_clone_written,
        "runtime_seconds": result.exact_runtime_seconds,
    }


def _aggregate_results_by_tau(
    tau_values: Sequence[float],
    process_results: Sequence[tuple[_PreparedProcessContext, Sequence[TauDiagnosticResult]]],
) -> list[dict[str, Any]]:
    aggregates: list[dict[str, Any]] = []
    for tau in tau_values:
        rows = [row for _, results in process_results for row in results if abs(row.tau - tau) <= 1e-15]
        selector_gaps = [row.gap_exchanges for row in rows if row.exact_selector_union_count is not None and row.gap_exchanges is not None]
        exact_solved = sum(1 for row in rows if row.exact_result_valid)
        def _is_skipped_status(status: str) -> bool:
            return status.startswith(("skipped", "limit_exceeded", "solver_unavailable"))
        skipped = sum(
            1
            for row in rows
            if (
                (row.sign_mode == SIGN_MODE_POSITIVE and _is_skipped_status(str(row.positive_status)))
                or (row.sign_mode == SIGN_MODE_NEGATIVE and _is_skipped_status(str(row.negative_status)))
                or (
                    row.sign_mode == SIGN_MODE_BOTH
                    and (
                        _is_skipped_status(str(row.positive_status))
                        or _is_skipped_status(str(row.negative_status))
                    )
                )
            )
        )
        invalid = sum(1 for row in rows if row.exact_solver_optimal and not row.exact_result_valid)
        greedy_equals_exact = sum(
            1
            for row in rows
            if row.exact_selector_union_count is not None and row.greedy_selector_union_count == row.exact_selector_union_count
        )
        exact_better = sum(
            1
            for row in rows
            if row.exact_selector_union_count is not None and row.exact_selector_union_count < row.greedy_selector_union_count
        )
        aggregates.append(
            {
                "tau": float(tau),
                "total_processes": len(rows),
                "exact_solved_processes": exact_solved,
                "skipped_processes": skipped,
                "invalid_exact_results": invalid,
                "greedy_equals_exact_count": greedy_equals_exact,
                "exact_better_count": exact_better,
                "max_selector_gap": None if not selector_gaps else int(max(selector_gaps)),
                "median_selector_gap": None if not selector_gaps else float(np.median(np.asarray(selector_gaps, dtype=float))),
                "total_greedy_selector_count": int(sum(row.greedy_selector_union_count for row in rows)),
                "total_exact_selector_count": int(sum(row.exact_selector_union_count or 0 for row in rows)),
            }
        )
    return aggregates


def run_greedy_exact_diagnostic(
    config: GreedyExactDiagnosticConfig,
    *,
    progress_callback: Optional[ProgressCallback] = None,
) -> GreedyExactDiagnosticResult:
    tau_values = _normalise_tau_values(config.tau_values)
    sign_mode = _normalise_sign_mode(config.sign_mode)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    multiple_tau = len(tau_values) > 1
    batch_mode = not str(config.process_query or "").strip()

    loaded = _load_diagnostic_inputs(config)
    process_entries = _resolve_process_entries(loaded.database_archive, config.process_query)
    total_steps = max(1, len(process_entries) * len(tau_values)) + 2
    _emit_progress(
        progress_callback,
        "Loading archive and diagnostic inputs...",
        1,
        total_steps,
    )

    warnings: List[Dict[str, str]] = []
    clones: List[Dict[str, Any]] = []
    debug_rows: List[Dict[str, Any]] = []
    process_result_groups: List[tuple[_PreparedProcessContext, List[TauDiagnosticResult]]] = []
    tau_results: List[TauDiagnosticResult] = []
    existing_process_names = {entry.name for entry in loaded.database_archive.processes.values()}
    created_at = datetime.now(timezone.utc).isoformat()
    step_index = 2
    for process_entry in process_entries:
        context = _prepare_context_for_entry(config, loaded, process_entry)
        process_tau_results: List[TauDiagnosticResult] = []
        for tau in tau_values:
            tau_result = _evaluate_tau(
                context,
                tau=tau,
                sign_mode=sign_mode,
                config=config,
                progress_callback=progress_callback,
                current=step_index,
                total=total_steps,
            )
            step_index += 1
            if not batch_mode:
                greedy_clone_name = _unique_clone_name(
                    _default_clone_name(context.process_name, "greedy", tau, multiple_tau=multiple_tau),
                    existing_process_names,
                    tau,
                )
                greedy_clone = _clone_process_with_selection(
                    context.process_data,
                    clone_name=greedy_clone_name,
                    clone_kind="greedy",
                    selected_mask=tau_result.greedy_union_mask,
                    context=context,
                    tau=tau,
                    sign_mode=sign_mode,
                    solver_type="deterministic_greedy_tau_cover",
                    created_at=created_at,
                )
                tau_result.greedy_clone_name = greedy_clone_name
                clones.append(greedy_clone)

                if tau_result.exact_result_valid and tau_result.exact_union_mask is not None:
                    exact_clone_name = _unique_clone_name(
                        _default_clone_name(context.process_name, "exact", tau, multiple_tau=multiple_tau),
                        existing_process_names,
                        tau,
                    )
                    exact_clone = _clone_process_with_selection(
                        context.process_data,
                        clone_name=exact_clone_name,
                        clone_kind="exact",
                        selected_mask=tau_result.exact_union_mask,
                        context=context,
                        tau=tau,
                        sign_mode=sign_mode,
                        solver_type="scipy.optimize.milp/HiGHS",
                        created_at=created_at,
                    )
                    tau_result.exact_clone_name = exact_clone_name
                    tau_result.exact_clone_written = True
                    clones.append(exact_clone)
                else:
                    warnings.append(
                        _metadata_warning(
                            f"Exact diagnostic clone was not written for process={context.process_id} tau={format(tau, '.15g')} "
                            f"because the exact result status was `{tau_result.exact_status}`."
                        )
                    )
                    debug_rows.extend(tau_result.exact_debug_rows)
            else:
                if not tau_result.exact_result_valid:
                    warnings.append(
                        _metadata_warning(
                            f"Exact diagnostic was not written for process={context.process_id} tau={format(tau, '.15g')} "
                            f"because batch mode records CSV/metadata only and the exact result status was `{tau_result.exact_status}`."
                        )
                    )
                    debug_rows.extend(tau_result.exact_debug_rows)
            process_tau_results.append(tau_result)
            tau_results.append(tau_result)
        process_result_groups.append((context, process_tau_results))

    diagnostic_zip = output_dir / DEFAULT_DIAGNOSTIC_ZIP_NAME if not batch_mode else None
    metadata_json = output_dir / DEFAULT_METADATA_NAME
    summary_csv = output_dir / DEFAULT_SUMMARY_CSV_NAME
    debug_csv = output_dir / DEFAULT_DEBUG_CSV_NAME if debug_rows else None

    _emit_progress(progress_callback, "Writing diagnostic artefacts...", total_steps - 1, total_steps)
    if diagnostic_zip is not None:
        single_context = process_result_groups[0][0]
        _write_diagnostic_zip(
            diagnostic_zip,
            database_source=config.database,
            selected_process_path=single_context.process_path,
            clones=clones,
        )
    _write_summary_csv(summary_csv, tau_results)
    if debug_csv is not None:
        _write_debug_csv(debug_csv, debug_rows)

    n_exact_clones_written = sum(1 for result in tau_results if result.exact_clone_written)
    all_greedy_certificates_pass = all(result.greedy_certificate_pass for result in tau_results)
    all_exact_written_certificates_pass = n_exact_clones_written > 0 and all(
        result.exact_certificate_pass for result in tau_results if result.exact_clone_written
    )
    all_exact_results_valid = all(result.exact_result_valid for result in tau_results)
    aggregate_by_tau = _aggregate_results_by_tau(tau_values, process_result_groups)
    selected_process_metadata = (
        {
            "mode": "all_processes",
            "process_count": len(process_entries),
        }
        if batch_mode
        else {
            "process_id": process_result_groups[0][0].process_id,
            "process_name": process_result_groups[0][0].process_name,
            "process_path": process_result_groups[0][0].process_path,
            "candidate_exchanges": process_result_groups[0][0].candidate_exchange_count,
            "protected_exchanges": process_result_groups[0][0].protected_exchange_count,
        }
    )
    metadata = {
        "warning_text": _WARNING_TEXT,
        "database": config.database,
        "methods": config.methods or "",
        "output_mode": "batch_csv_metadata_only" if batch_mode else "complete_database_with_diagnostic_processes",
        "batch_mode": batch_mode,
        "lcia_method_source": loaded.lcia_method_source,
        "internal_lcia_methods_ignored": loaded.internal_lcia_methods_ignored,
        "method_selection": config.method_selection,
        "allow_water_mass_volume_override": config.allow_water_mass_volume_override,
        "selected_process": selected_process_metadata,
        "tau_values": [float(value) for value in tau_values],
        "sign_mode": sign_mode,
        "solver_limits": {
            "max_binary_variables": config.max_binary_variables,
            "max_active_rows": config.max_active_rows,
            "solver_time_limit_seconds": config.solver_time_limit_seconds,
            "max_tau_values": DEFAULT_MAX_TAU_VALUES,
        },
        "solver": {
            "scipy_milp_available": SCIPY_MILP_AVAILABLE,
        },
        "selected_lcia_categories": [
            {
                "category_id": category.category_id,
                "category_name": category.name,
                "method_id": category.method_id,
                "method_name": category.method_name,
            }
            for category in loaded.selected_categories
        ],
        "empty_selected_lcia_categories": [
            {
                "category_id": getattr(category, "category_id", ""),
                "category_name": getattr(category, "category_name", ""),
                "method_id": getattr(category, "method_id", ""),
                "method_name": getattr(category, "method_name", ""),
            }
            for category in loaded.empty_selected_categories
        ],
        "artifacts": {
            "metadata_json": str(metadata_json),
            "summary_csv": str(summary_csv),
            **({"diagnostic_zip": str(diagnostic_zip)} if diagnostic_zip is not None else {}),
            **({"debug_csv": str(debug_csv)} if debug_csv is not None else {}),
        },
        "summary": {
            "n_processes": len(process_entries),
            "n_tau_values": len(tau_values),
            "tau_values_text": ", ".join(format(value, ".15g") for value in tau_values),
            "n_greedy_clones_written": 0 if batch_mode else len(tau_values),
            "n_exact_clones_written": n_exact_clones_written,
            "all_greedy_certificates_pass": all_greedy_certificates_pass,
            "all_exact_written_certificates_pass": all_exact_written_certificates_pass,
            "all_exact_tau_optimal": all(result.exact_solver_optimal for result in tau_results),
            "all_exact_results_valid": all_exact_results_valid,
            "core_artifact_count": 2 if batch_mode else 3,
        },
        "aggregate_by_tau": aggregate_by_tau,
        **(
            {
                "tau_results": [
                    _serialise_tau_result(process_result_groups[0][0], result)
                    for result in process_result_groups[0][1]
                ]
            }
            if not batch_mode
            else {
                "process_results": [
                    {
                        "process_id": context.process_id,
                        "process_name": context.process_name,
                        "process_path": context.process_path,
                        "candidate_exchanges": context.candidate_exchange_count,
                        "protected_exchanges": context.protected_exchange_count,
                        "tau_results": [_serialise_tau_result(context, result) for result in results],
                    }
                    for context, results in process_result_groups
                ]
            }
        ),
        "warnings": warnings,
    }
    metadata_json.write_text(json.dumps(metadata, indent=2, ensure_ascii=True), encoding="utf-8")

    _emit_progress(progress_callback, "Diagnostic artefacts ready.", total_steps, total_steps)
    return GreedyExactDiagnosticResult(
        diagnostic_zip=str(diagnostic_zip) if diagnostic_zip is not None else None,
        metadata_json=str(metadata_json),
        summary_csv=str(summary_csv),
        debug_csv=str(debug_csv) if debug_csv is not None else None,
        metadata=metadata,
    )
def diagnostic_warning_text() -> str:
    return _WARNING_TEXT


def normalise_diagnostic_tau_values(values: Sequence[float]) -> List[float]:
    return _normalise_tau_values(values)
