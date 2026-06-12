import csv
import json
import os
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mplconfig")

import lci_reduce.greedy_exact_diagnostic as ged
from lci_reduce import reducer as reducer_module
from lci_reduce.contribution import _CategoryRowSet
from lci_reduce.greedy_exact_diagnostic import GreedyExactDiagnosticConfig
from lci_reduce.archive_reader import parse_json_object
from lci_reduce.sparse_cover import build_sparse_cover_matrix, greedy_tau_cover_sparse


def _patch_fake_milp(monkeypatch: pytest.MonkeyPatch) -> None:
    def _csr_matrix(value):
        return np.asarray(value, dtype=float)

    def _linear_constraint(matrix, lb, ub):
        return SimpleNamespace(A=np.asarray(matrix, dtype=float), lb=np.asarray(lb, dtype=float), ub=np.asarray(ub, dtype=float))

    def _bounds(lb, ub):
        return SimpleNamespace(lb=np.asarray(lb, dtype=float), ub=np.asarray(ub, dtype=float))

    def _milp(*, c, constraints, integrality, bounds, options):
        matrix = np.asarray(constraints.A, dtype=float)
        lower = np.asarray(constraints.lb, dtype=float)
        n_vars = int(len(c))
        best_x = None
        best_objective = None
        lower_bounds = np.asarray(bounds.lb, dtype=float)
        upper_bounds = np.asarray(bounds.ub, dtype=float)
        for mask_int in range(1 << n_vars):
            x = np.array([(mask_int >> index) & 1 for index in range(n_vars)], dtype=float)
            if np.any(x < lower_bounds - 1e-12) or np.any(x > upper_bounds + 1e-12):
                continue
            if np.all(matrix @ x + 1e-12 >= lower):
                objective = float(np.asarray(c, dtype=float) @ x)
                if best_objective is None or objective < best_objective - 1e-12:
                    best_objective = objective
                    best_x = x
        if best_x is None:
            return SimpleNamespace(status=2, message="infeasible", x=None)
        return SimpleNamespace(status=0, message="optimal", x=best_x)

    monkeypatch.setattr(ged, "SCIPY_MILP_AVAILABLE", True)
    monkeypatch.setattr(ged, "sparse", SimpleNamespace(csr_matrix=_csr_matrix))
    monkeypatch.setattr(ged, "LinearConstraint", _linear_constraint)
    monkeypatch.setattr(ged, "Bounds", _bounds)
    monkeypatch.setattr(ged, "milp", _milp)


def _build_model(matrix: np.ndarray):
    rowsets = []
    for row_index, row in enumerate(np.asarray(matrix, dtype=float)):
        exact_items = tuple((column_index, float(value)) for column_index, value in enumerate(row) if abs(float(value)) > 1e-12)
        rowsets.append(
            _CategoryRowSet(
                exact_items=exact_items,
                scenario_rows=(tuple(),),
                metadata_rows=tuple(),
            )
        )
    return build_sparse_cover_matrix(rowsets, np.asarray(matrix).shape[1], positive=True, tol=1e-12)


def _build_mixed_model():
    return build_sparse_cover_matrix(
        (
            _CategoryRowSet(
                exact_items=((0, 2.0),),
                scenario_rows=(((1, 1.5),), ((2, 0.5),)),
                metadata_rows=tuple(),
            ),
        ),
        3,
        positive=True,
        tol=1e-12,
    )


def _make_diagnostic_database_zip(base: Path, name: str = "diagnostic_db.zip") -> Path:
    database_zip = base / name
    files = {
        "processes/process-1.json": {
            "@id": "process-1",
            "@type": "Process",
            "name": "Diagnostic Process",
            "description": "Original process description.",
            "exchanges": [
                {
                    "@id": "product",
                    "amount": 1.0,
                    "flow": {"@id": "flow-product", "name": "Product"},
                    "unit": {"name": "kg"},
                    "quantitativeReference": True,
                },
                {
                    "@id": "elem-co2",
                    "amount": 10.0,
                    "flow": {"@id": "flow-co2", "name": "CO2"},
                    "unit": {"name": "kg"},
                },
                {
                    "@id": "elem-ch4",
                    "amount": 0.1,
                    "flow": {"@id": "flow-ch4", "name": "CH4"},
                    "unit": {"name": "kg"},
                },
                {
                    "@id": "elem-unchar",
                    "amount": 0.5,
                    "flow": {"@id": "flow-n2o", "name": "N2O"},
                    "unit": {"name": "kg"},
                },
            ],
        },
        "flows/flow-product.json": {
            "@id": "flow-product",
            "@type": "Flow",
            "name": "Product",
            "flowType": "PRODUCT_FLOW",
            "categoryPath": "products",
        },
        "flows/flow-co2.json": {
            "@id": "flow-co2",
            "@type": "Flow",
            "name": "CO2",
            "flowType": "ELEMENTARY_FLOW",
            "categoryPath": "air/urban air",
        },
        "flows/flow-ch4.json": {
            "@id": "flow-ch4",
            "@type": "Flow",
            "name": "CH4",
            "flowType": "ELEMENTARY_FLOW",
            "categoryPath": "air/urban air",
        },
        "flows/flow-n2o.json": {
            "@id": "flow-n2o",
            "@type": "Flow",
            "name": "N2O",
            "flowType": "ELEMENTARY_FLOW",
            "categoryPath": "air/urban air",
        },
        "lcia_methods/method-1.json": {
            "@id": "method-1",
            "@type": "ImpactMethod",
            "name": "IPCC 2021",
        },
        "lcia_categories/category-1.json": {
            "@id": "category-1",
            "@type": "ImpactCategory",
            "name": "Climate change",
            "impactMethod": {"@id": "method-1", "name": "IPCC 2021"},
            "referenceUnitName": "kg",
            "impactFactors": [
                {"flow": {"@id": "flow-co2", "name": "CO2"}, "value": 10.0, "unitName": "kg"},
                {"flow": {"@id": "flow-ch4", "name": "CH4"}, "value": 0.1, "unitName": "kg"},
            ],
        },
    }
    with zipfile.ZipFile(database_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for rel_path, data in files.items():
            archive.writestr(rel_path, json.dumps(data))
    return database_zip


def _make_batch_diagnostic_database_zip(base: Path, name: str = "diagnostic_db_batch.zip") -> Path:
    database_zip = base / name
    files = {
        "processes/process-1.json": {
            "@id": "process-1",
            "@type": "Process",
            "name": "Diagnostic Process A",
            "exchanges": [
                {"@id": "product-a", "amount": 1.0, "flow": {"@id": "flow-product-a", "name": "Product A"}, "unit": {"name": "kg"}, "quantitativeReference": True},
                {"@id": "elem-a1", "amount": 5.0, "flow": {"@id": "flow-co2", "name": "CO2"}, "unit": {"name": "kg"}},
                {"@id": "elem-a2", "amount": 2.0, "flow": {"@id": "flow-ch4", "name": "CH4"}, "unit": {"name": "kg"}},
            ],
        },
        "processes/process-2.json": {
            "@id": "process-2",
            "@type": "Process",
            "name": "Diagnostic Process B",
            "exchanges": [
                {"@id": "product-b", "amount": 1.0, "flow": {"@id": "flow-product-b", "name": "Product B"}, "unit": {"name": "kg"}, "quantitativeReference": True},
                {"@id": "elem-b1", "amount": 1.5, "flow": {"@id": "flow-co2", "name": "CO2"}, "unit": {"name": "kg"}},
                {"@id": "elem-b2", "amount": 8.0, "flow": {"@id": "flow-ch4", "name": "CH4"}, "unit": {"name": "kg"}},
            ],
        },
        "flows/flow-product-a.json": {"@id": "flow-product-a", "@type": "Flow", "name": "Product A", "flowType": "PRODUCT_FLOW", "categoryPath": "products"},
        "flows/flow-product-b.json": {"@id": "flow-product-b", "@type": "Flow", "name": "Product B", "flowType": "PRODUCT_FLOW", "categoryPath": "products"},
        "flows/flow-co2.json": {"@id": "flow-co2", "@type": "Flow", "name": "CO2", "flowType": "ELEMENTARY_FLOW", "categoryPath": "air/urban air"},
        "flows/flow-ch4.json": {"@id": "flow-ch4", "@type": "Flow", "name": "CH4", "flowType": "ELEMENTARY_FLOW", "categoryPath": "air/urban air"},
        "lcia_methods/method-1.json": {"@id": "method-1", "@type": "ImpactMethod", "name": "IPCC 2021"},
        "lcia_categories/category-1.json": {
            "@id": "category-1",
            "@type": "ImpactCategory",
            "name": "Climate change",
            "impactMethod": {"@id": "method-1", "name": "IPCC 2021"},
            "referenceUnitName": "kg",
            "impactFactors": [
                {"flow": {"@id": "flow-co2", "name": "CO2"}, "value": 1.0, "unitName": "kg"},
                {"flow": {"@id": "flow-ch4", "name": "CH4"}, "value": 0.5, "unitName": "kg"},
            ],
        },
    }
    with zipfile.ZipFile(database_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for rel_path, data in files.items():
            archive.writestr(rel_path, json.dumps(data))
    return database_zip


def test_exact_solver_matches_greedy_on_trivial_one_row_case(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_fake_milp(monkeypatch)
    model = _build_model(np.array([[5.0, 4.0, 1.0]], dtype=float))

    greedy = greedy_tau_cover_sparse(model, 0.9, exchange_keys=["a", "b", "c"], tol=1e-12)
    exact = ged._solve_exact_cover(
        model,
        tau=0.9,
        tolerance=1e-12,
        max_binary_variables=250,
        max_active_rows=1000,
        solver_time_limit_seconds=30.0,
    )

    assert exact.is_optimal
    assert exact.result_valid is True
    assert int(greedy.sum()) == 2
    assert int(exact.selected_mask.sum()) == 2


def test_exact_solver_improves_on_greedy_for_known_counterexample(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_fake_milp(monkeypatch)
    model = _build_model(np.array([[0.0, 2.0, 3.0], [5.0, 3.0, 0.0]], dtype=float))

    greedy = greedy_tau_cover_sparse(model, 0.6, exchange_keys=["a", "b", "c"], tol=1e-12)
    exact = ged._solve_exact_cover(
        model,
        tau=0.6,
        tolerance=1e-12,
        max_binary_variables=250,
        max_active_rows=1000,
        solver_time_limit_seconds=30.0,
    )

    assert int(greedy.sum()) == 3
    assert exact.is_optimal
    assert exact.result_valid is True
    assert int(exact.selected_mask.sum()) == 2
    assert exact.selected_mask.tolist() == [True, False, True]


def test_exact_solver_handles_positive_and_negative_models_separately(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_fake_milp(monkeypatch)
    contributions = np.array([[4.0, -5.0, 0.0], [0.0, -1.0, 3.0]], dtype=float)
    positive_model = _build_model(np.maximum(contributions, 0.0))
    negative_model = _build_model(np.maximum(-contributions, 0.0))

    positive = ged._solve_exact_cover(
        positive_model,
        tau=0.9,
        tolerance=1e-12,
        max_binary_variables=250,
        max_active_rows=1000,
        solver_time_limit_seconds=30.0,
    )
    negative = ged._solve_exact_cover(
        negative_model,
        tau=0.9,
        tolerance=1e-12,
        max_binary_variables=250,
        max_active_rows=1000,
        solver_time_limit_seconds=30.0,
    )

    assert positive.is_optimal
    assert negative.is_optimal
    assert positive.result_valid is True
    assert negative.result_valid is True
    assert positive.selected_mask.tolist() == [True, False, True]
    assert negative.selected_mask.tolist() == [False, True, False]
    assert (positive.selected_mask | negative.selected_mask).tolist() == [True, True, True]


def test_dense_active_matrix_matches_retained_by_row_for_deterministic_masks() -> None:
    model = _build_mixed_model()
    matrix = ged._dense_active_matrix(model)
    message = ged._matrix_equivalence_message(
        model,
        active_matrix=matrix,
        protected_mask=np.array([False, True, False], dtype=bool),
        tolerance=1e-12,
        extra_masks=(np.array([True, False, False], dtype=bool),),
    )
    assert message is None


def test_exact_solver_uses_normalised_constraints_for_tiny_row(monkeypatch: pytest.MonkeyPatch) -> None:
    def _csr_matrix(value):
        return np.asarray(value, dtype=float)

    def _linear_constraint(matrix, lb, ub):
        return SimpleNamespace(A=np.asarray(matrix, dtype=float), lb=np.asarray(lb, dtype=float), ub=np.asarray(ub, dtype=float))

    def _bounds(lb, ub):
        return SimpleNamespace(lb=np.asarray(lb, dtype=float), ub=np.asarray(ub, dtype=float))

    def _milp(*, c, constraints, integrality, bounds, options):
        lower = np.asarray(constraints.lb, dtype=float)
        if float(lower.min()) < 1e-8:
            return SimpleNamespace(status=0, message="optimal", x=np.zeros(len(c), dtype=float))
        matrix = np.asarray(constraints.A, dtype=float)
        best_x = None
        best_objective = None
        lower_bounds = np.asarray(bounds.lb, dtype=float)
        upper_bounds = np.asarray(bounds.ub, dtype=float)
        for mask_int in range(1 << len(c)):
            x = np.array([(mask_int >> index) & 1 for index in range(len(c))], dtype=float)
            if np.any(x < lower_bounds - 1e-12) or np.any(x > upper_bounds + 1e-12):
                continue
            if np.all(matrix @ x + 1e-12 >= lower):
                objective = float(np.asarray(c, dtype=float) @ x)
                if best_objective is None or objective < best_objective - 1e-12:
                    best_objective = objective
                    best_x = x
        return SimpleNamespace(status=0, message="optimal", x=best_x)

    monkeypatch.setattr(ged, "SCIPY_MILP_AVAILABLE", True)
    monkeypatch.setattr(ged, "sparse", SimpleNamespace(csr_matrix=_csr_matrix))
    monkeypatch.setattr(ged, "LinearConstraint", _linear_constraint)
    monkeypatch.setattr(ged, "Bounds", _bounds)
    monkeypatch.setattr(ged, "milp", _milp)

    model = _build_model(np.array([[6.9e-7, 0.0], [0.0, 1.0]], dtype=float))
    exact = ged._solve_exact_cover(
        model,
        tau=0.95,
        tolerance=1e-12,
        max_binary_variables=250,
        max_active_rows=1000,
        solver_time_limit_seconds=30.0,
    )

    assert exact.is_optimal
    assert exact.result_valid is True
    assert exact.selected_mask.tolist() == [True, True]


def test_exact_solver_handles_widely_different_row_scales(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_fake_milp(monkeypatch)
    model = _build_model(np.array([[1.0, 0.0, 0.4], [0.0, 1.0e-6, 0.6e-6], [0.0, 0.0, 1.0e-9]], dtype=float))
    exact = ged._solve_exact_cover(
        model,
        tau=0.95,
        tolerance=1e-12,
        max_binary_variables=250,
        max_active_rows=1000,
        solver_time_limit_seconds=30.0,
    )

    assert exact.is_optimal
    assert exact.result_valid is True
    assert exact.verifier_min_coverage is not None
    assert exact.solver_min_coverage is not None
    assert exact.verifier_min_coverage >= 0.95
    assert exact.solver_min_coverage >= 0.95


def test_prepare_sparse_reduction_context_does_not_change_normal_reducer_output(tmp_path: Path) -> None:
    database_zip = _make_diagnostic_database_zip(tmp_path, name="diagnostic_db_reducer_consistency.zip")
    archive = ged.load_archive(str(database_zip), require_processes=True, require_flows=True, conversion_scope="inspect")
    process_data = archive.processes["process-1"].data
    categories = ged.collect_categories((archive,), diagnostic_file="test")

    result = reducer_module.reduce_process(
        process_data=process_data,
        flow_lookup=archive.flows,
        categories=categories,
        tau=0.95,
        uncharacterised_policy="keep",
        strict_units=True,
        tol=1e-12,
        database_name="test-db",
        unit_registry=archive.units,
        return_payload="full",
    )

    assert result.process_row["positive_cover_ok"] is True
    assert result.process_row["negative_cover_ok"] is True
    assert result.process_row["n_protected_exchanges"] == 1


def test_run_greedy_exact_diagnostic_preserves_original_and_writes_clones(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_fake_milp(monkeypatch)
    database_zip = _make_diagnostic_database_zip(tmp_path)
    output_dir = tmp_path / "out"
    result = ged.run_greedy_exact_diagnostic(
        GreedyExactDiagnosticConfig(
            database=str(database_zip),
            methods=None,
            output_dir=str(output_dir),
            method_selection="all",
            process_query="process-1",
            tau_values=[0.95],
            sign_mode="both",
            strict_units=True,
            tolerance=1e-12,
        )
    )

    assert result.metadata["summary"]["all_greedy_certificates_pass"] is True
    assert result.metadata["summary"]["all_exact_written_certificates_pass"] is True
    assert result.metadata["summary"]["all_exact_results_valid"] is True
    assert result.metadata["summary"]["n_exact_clones_written"] == 1
    assert result.metadata["tau_results"][0]["exact_min_coverage"] > 0.95
    selector_comparison = result.metadata["tau_results"][0]["selector_comparison"]
    assert selector_comparison["positive"]["jaccard"] is not None
    assert "greedy_export" in result.metadata["tau_results"][0]["selected_exchange_ids"]
    assert result.metadata["tau_results"][0]["exact_positive_solver_min_coverage"] is not None
    assert result.metadata["tau_results"][0]["exact_positive_verifier_min_coverage"] is not None

    with zipfile.ZipFile(database_zip, "r") as original_archive:
        original_entries = {name: original_archive.read(name) for name in original_archive.namelist()}
    with zipfile.ZipFile(result.diagnostic_zip, "r") as diagnostic_archive:
        diagnostic_entries = {name: diagnostic_archive.read(name) for name in diagnostic_archive.namelist()}

    assert diagnostic_entries["processes/process-1.json"] == original_entries["processes/process-1.json"]
    for name, payload in original_entries.items():
        if name.startswith("processes/"):
            continue
        assert diagnostic_entries[name] == payload

    extra_processes = sorted(set(diagnostic_entries) - set(original_entries))
    assert len(extra_processes) == 2
    parsed = [json.loads(diagnostic_entries[name].decode("utf-8")) for name in extra_processes]
    clone_names = {item["name"] for item in parsed}
    clone_ids = {item["@id"] for item in parsed}
    assert "Diagnostic Process_greedy" in clone_names
    assert "Diagnostic Process_exact" in clone_names
    assert "process-1" not in clone_ids
    assert result.metadata["selected_process"]["protected_exchanges"] == 1
    assert set(result.metadata["artifacts"]) == {"diagnostic_zip", "metadata_json", "summary_csv"}

    for name, payload in diagnostic_entries.items():
        if not name.lower().endswith(".json"):
            continue
        parse_json_object(payload, name)


def test_exact_failure_does_not_create_fake_exact_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ged, "SCIPY_MILP_AVAILABLE", False)
    database_zip = _make_diagnostic_database_zip(tmp_path, name="diagnostic_db_failure.zip")
    output_dir = tmp_path / "out_failure"
    result = ged.run_greedy_exact_diagnostic(
        GreedyExactDiagnosticConfig(
            database=str(database_zip),
            methods=None,
            output_dir=str(output_dir),
            method_selection="all",
            process_query="Diagnostic Process",
            tau_values=[0.95],
            sign_mode="both",
            strict_units=True,
            tolerance=1e-12,
        )
    )

    assert result.metadata["summary"]["n_exact_clones_written"] == 0
    assert result.metadata["summary"]["all_exact_results_valid"] is False
    assert any("exact diagnostic clone was not written" in warning["message"].lower() for warning in result.metadata["warnings"])
    assert any(item["exact_union_count"] is None for item in result.metadata["tau_results"])

    with zipfile.ZipFile(database_zip, "r") as original_archive:
        original_process_names = {name for name in original_archive.namelist() if name.startswith("processes/")}
    with zipfile.ZipFile(result.diagnostic_zip, "r") as diagnostic_archive:
        diagnostic_process_names = {name for name in diagnostic_archive.namelist() if name.startswith("processes/")}
        extra_processes = diagnostic_process_names - original_process_names
        assert len(extra_processes) == 1
        clone = json.loads(diagnostic_archive.read(next(iter(extra_processes))).decode("utf-8"))
    assert clone["name"] == "Diagnostic Process_greedy"


def test_summary_csv_contains_one_row_per_tau_and_multiple_clone_pairs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_fake_milp(monkeypatch)
    database_zip = _make_diagnostic_database_zip(tmp_path, name="diagnostic_db_curve.zip")
    output_dir = tmp_path / "out_curve"
    result = ged.run_greedy_exact_diagnostic(
        GreedyExactDiagnosticConfig(
            database=str(database_zip),
            methods=None,
            output_dir=str(output_dir),
            method_selection="all",
            process_query="process-1",
            tau_values=[0.6, 0.7, 0.8, 0.9, 0.95],
            sign_mode="both",
            strict_units=True,
            tolerance=1e-12,
        )
    )

    with open(result.summary_csv, "r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 5
    assert "positive_jaccard" in rows[0]
    assert "union_only_exact" in rows[0]
    with zipfile.ZipFile(result.diagnostic_zip, "r") as diagnostic_archive:
        process_entries = [name for name in diagnostic_archive.namelist() if name.startswith("processes/")]
    assert len(process_entries) == 1 + (2 * 5)


def test_solver_optimal_but_invalid_certificate_does_not_write_exact_clone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_fake_milp(monkeypatch)

    def _invalid_milp(*, c, constraints, integrality, bounds, options):
        x = np.asarray(bounds.lb, dtype=float).copy()
        return SimpleNamespace(status=0, message="optimal", x=x)

    monkeypatch.setattr(ged, "milp", _invalid_milp)
    database_zip = _make_diagnostic_database_zip(tmp_path, name="diagnostic_db_invalid_certificate.zip")
    output_dir = tmp_path / "out_invalid_certificate"
    result = ged.run_greedy_exact_diagnostic(
        GreedyExactDiagnosticConfig(
            database=str(database_zip),
            methods=None,
            output_dir=str(output_dir),
            method_selection="all",
            process_query="process-1",
            tau_values=[0.95],
            sign_mode="positive",
            strict_units=True,
            tolerance=1e-12,
        )
    )

    row = result.metadata["tau_results"][0]
    assert row["exact_solver_optimal"] is True
    assert row["exact_result_valid"] is False
    assert row["exact_status"].startswith("invalid_exact_certificate")
    assert row["exact_union_count"] is None
    assert result.metadata["summary"]["n_exact_clones_written"] == 0


def test_solver_optimal_but_dominance_failure_does_not_write_exact_clone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_fake_milp(monkeypatch)

    def _dominance_milp(*, c, constraints, integrality, bounds, options):
        x = np.asarray(bounds.lb, dtype=float).copy()
        x[np.asarray(bounds.ub, dtype=float) > 0.5] = 1.0
        return SimpleNamespace(status=0, message="optimal", x=x)

    monkeypatch.setattr(ged, "milp", _dominance_milp)
    database_zip = _make_diagnostic_database_zip(tmp_path, name="diagnostic_db_invalid_dominance.zip")
    output_dir = tmp_path / "out_invalid_dominance"
    result = ged.run_greedy_exact_diagnostic(
        GreedyExactDiagnosticConfig(
            database=str(database_zip),
            methods=None,
            output_dir=str(output_dir),
            method_selection="all",
            process_query="process-1",
            tau_values=[0.95],
            sign_mode="positive",
            strict_units=True,
            tolerance=1e-12,
        )
    )

    row = result.metadata["tau_results"][0]
    assert row["exact_solver_optimal"] is True
    assert row["exact_result_valid"] is False
    assert row["exact_status"].startswith("invalid_exact_dominance")
    assert row["exact_union_count"] is None
    assert result.metadata["summary"]["n_exact_clones_written"] == 0
    with open(result.summary_csv, "r", encoding="utf-8", newline="") as handle:
        summary_rows = list(csv.DictReader(handle))
    assert summary_rows[0]["exact_exported_elementary_count"] == ""
    assert summary_rows[0]["gap_exchanges"] == ""
    assert result.debug_csv is not None


def test_empty_process_query_runs_batch_mode_over_all_processes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_fake_milp(monkeypatch)
    database_zip = _make_batch_diagnostic_database_zip(tmp_path)
    output_dir = tmp_path / "out_batch"

    result = ged.run_greedy_exact_diagnostic(
        GreedyExactDiagnosticConfig(
            database=str(database_zip),
            methods=None,
            output_dir=str(output_dir),
            method_selection="all",
            process_query="",
            tau_values=[0.8, 0.95],
            sign_mode="both",
            strict_units=True,
            tolerance=1e-12,
        )
    )

    assert result.diagnostic_zip is None
    assert result.metadata["batch_mode"] is True
    assert result.metadata["output_mode"] == "batch_csv_metadata_only"
    assert result.metadata["selected_process"]["mode"] == "all_processes"
    assert result.metadata["summary"]["n_processes"] == 2
    assert len(result.metadata["process_results"]) == 2
    assert len(result.metadata["aggregate_by_tau"]) == 2
    assert all(len(item["tau_results"]) == 2 for item in result.metadata["process_results"])
    with open(result.summary_csv, "r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert {row["process_id"] for row in rows} == {"process-1", "process-2"}
    assert not (output_dir / "greedy_exact_diagnostic.zip").exists()


def test_non_empty_process_query_preserves_single_process_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_fake_milp(monkeypatch)
    database_zip = _make_batch_diagnostic_database_zip(tmp_path, name="diagnostic_db_single.zip")
    output_dir = tmp_path / "out_single"

    result = ged.run_greedy_exact_diagnostic(
        GreedyExactDiagnosticConfig(
            database=str(database_zip),
            methods=None,
            output_dir=str(output_dir),
            method_selection="all",
            process_query="process-1",
            tau_values=[0.8],
            sign_mode="both",
            strict_units=True,
            tolerance=1e-12,
        )
    )

    assert result.diagnostic_zip is not None
    assert result.metadata["batch_mode"] is False
    assert result.metadata["selected_process"]["process_id"] == "process-1"
    assert "tau_results" in result.metadata
    assert "process_results" not in result.metadata
    with open(result.summary_csv, "r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["process_id"] == "process-1"


def test_oversized_process_is_skipped_cleanly_in_batch_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_fake_milp(monkeypatch)
    database_zip = _make_batch_diagnostic_database_zip(tmp_path, name="diagnostic_db_skip.zip")
    output_dir = tmp_path / "out_skip"

    result = ged.run_greedy_exact_diagnostic(
        GreedyExactDiagnosticConfig(
            database=str(database_zip),
            methods=None,
            output_dir=str(output_dir),
            method_selection="all",
            process_query="",
            tau_values=[0.8],
            sign_mode="both",
            strict_units=True,
            tolerance=1e-12,
            max_binary_variables=1,
        )
    )

    rows = result.metadata["process_results"]
    assert len(rows) == 2
    first_tau = rows[0]["tau_results"][0]
    assert first_tau["greedy_certificate_pass"] is True
    assert first_tau["exact_result_valid"] is False
    assert "limit_exceeded_binary_variables" in first_tau["exact_status"]
    aggregate = result.metadata["aggregate_by_tau"][0]
    assert aggregate["skipped_processes"] == 2
    assert aggregate["exact_solved_processes"] == 0


def test_batch_mode_does_not_write_diagnostic_process_clones(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_fake_milp(monkeypatch)
    database_zip = _make_batch_diagnostic_database_zip(tmp_path, name="diagnostic_db_no_clones.zip")
    output_dir = tmp_path / "out_no_clones"

    result = ged.run_greedy_exact_diagnostic(
        GreedyExactDiagnosticConfig(
            database=str(database_zip),
            methods=None,
            output_dir=str(output_dir),
            method_selection="all",
            process_query="",
            tau_values=[0.95],
            sign_mode="both",
            strict_units=True,
            tolerance=1e-12,
        )
    )

    assert result.diagnostic_zip is None
    assert result.metadata["summary"]["n_greedy_clones_written"] == 0
    assert result.metadata["summary"]["n_exact_clones_written"] == 0
    assert "diagnostic_zip" not in result.metadata["artifacts"]
