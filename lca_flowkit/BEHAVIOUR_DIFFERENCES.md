# Behaviour Differences

This file records intentional differences between the legacy `lci_reduce` implementation and the clean package rewrite.

## 1. Package surface

Old behaviour:
- workflows were exposed through CLI and GUI oriented modules;
- analysis helpers were intertwined with GUI expectations.

New behaviour:
- the clean package exposes `reduce_database`, `create_priority_file`, and `analyse_flows` directly;
- results are returned as structured Python objects plus sidecar files.

Reason:
- package-first design for scripts, notebooks, tests, and publication review.

Change type:
- simplification.

## 2. Output set

Old behaviour:
- the legacy reducer can emit GUI-facing and report-oriented artefacts, including a PDF workflow.

New behaviour:
- the clean package writes the auditable scientific core only:
  reduced ZIP, manifests, metadata, validation, and warnings for reduction;
  compact priority CSV, metadata, and warnings for priority generation.

Reason:
- the rewrite deliberately avoids GUI/backend coupling and keeps the implementation smaller.

Change type:
- simplification.

## 3. CF ambiguity handling

Old behaviour:
- ambiguity handling is spread across several modules, with sparse-cover optimisations and detailed GUI/CLI diagnostics.

New behaviour:
- the clean package builds explicit exact, finite, and regional scenario rows in one dedicated module;
- admissible regional CF ambiguity is always represented explicitly in coverage rows;
- if expansion becomes too large, the clean package raises `ScenarioExpansionError` instead of choosing one regional CF implicitly.

Reason:
- concentrate the scientific rule set in one auditable implementation path and avoid a second, deterministic regional fallback mode.

Change type:
- scientific modelling clarification.

## 4. EcoSpold1 flow matching

Old behaviour:
- EcoSpold1 support exists in the legacy implementation, but the clean rewrite uses an independently defined stable-ID scheme.

New behaviour:
- the clean package hashes EcoSpold1 method and process elementary flows with the same stable identity tuple:
  flow name, category path, and flow type.

Reason:
- the rewrite must match LCIA factors to process flows without reusing the old parser directly.

Change type:
- independent reimplementation.

## 5. Exact grouped-flow analysis

Old behaviour:
- grouped-flow analysis in the compact analyser is already conservative.

New behaviour:
- the clean analyser explicitly returns lower and upper conservative grouped bounds and always warns when a grouped result is not exact.

Reason:
- make the guarantee boundary explicit in notebook/script usage.

Change type:
- clarification.

## 6. Analysis result filtering

Old behaviour:
- the clean analyser accepted `result_types`, but the argument did not materially change the returned payload.

New behaviour:
- `result_types` now acts as a real output-section filter for `eta_bounds`, `loss_max_flows`, `witness_processes`, `top_n_repair`, `least_n_repair`, and `coverage_summary`;
- request bookkeeping such as matched/unmatched flows is always returned.

Reason:
- remove a misleading public argument and make the package API explicit and testable.

Change type:
- bug fix.
