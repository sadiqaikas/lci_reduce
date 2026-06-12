# Implementation Audit

## Files created

- `lca_flowkit/__init__.py`
- `lca_flowkit/errors.py`
- `lca_flowkit/models.py`
- `lca_flowkit/utils.py`
- `lca_flowkit/method_selection.py`
- `lca_flowkit/jsonld.py`
- `lca_flowkit/ecospold1.py`
- `lca_flowkit/inputs.py`
- `lca_flowkit/units.py`
- `lca_flowkit/scenarios.py`
- `lca_flowkit/cover.py`
- `lca_flowkit/reducer.py`
- `lca_flowkit/priority.py`
- `lca_flowkit/analysis.py`
- `lca_flowkit/reporting.py`
- `lca_flowkit/writers.py`
- `lca_flowkit/README.md`
- `lca_flowkit/BEHAVIOUR_DIFFERENCES.md`
- `lca_flowkit/IMPLEMENTATION_AUDIT.md`
- `tests/test_lca_flowkit_cf.py`
- `tests/test_lca_flowkit_analysis.py`
- `tests/test_lca_flowkit_workflows.py`

## Package structure

- `jsonld.py` and `ecospold1.py` are independent readers.
- `inputs.py` selects the reader and merges optional method inputs.
- `units.py` handles strict unit compatibility.
- `scenarios.py` builds exact, finite, and regional coverage rows.
- `cover.py` implements deterministic weighted greedy signed tau-cover and greedy ladders.
- `reducer.py` is the reduced-database public API.
- `priority.py` is the compact priority-file public API.
- `analysis.py` is the notebook/script analyser API.
- `reporting.py` and `writers.py` handle sidecar outputs.

## Public APIs

- `reduce_database(...)`
- `create_priority_file(...)`
- `analyse_flows(...)`

Each API returns structured Python data in addition to writing sidecar files.

## Tests added

The clean package adds controlled coverage for:

- exact CF matching;
- strict unit filtering;
- duplicate CF collapse;
- blank-location generic fallback;
- real regional CF ambiguity;
- scenario-expansion failure when regional rows exceed the configured limit;
- uncharacterised elementary-flow retention in the reducer;
- malformed EcoSpold1 XML handling;
- grouped-flow analyser conservative bounds;
- `analyse_flows(result_types=...)` output filtering;
- `analyse_flows(all_flows=True)` whole-file selection and selector validation;
- basic JSON-LD parity checks against the legacy implementation;
- EcoSpold1 reduction and priority-file generation.

## Performance choices

- matrices are built per process, not for the whole database at once;
- the greedy selector uses dense NumPy arrays for readability and deterministic behaviour;
- regional ambiguity uses a union-of-locations table with within-location products, not a full cross-product of regional cases;
- only metadata-equivalent CF candidates are collapsed before scenario expansion;
- compact priority analysis stays ledger-free and uses conservative grouped bounds.

## Behaviour copied from the existing implementation

- characterised contributions drive both reduction and priority ranking;
- positive and negative contributions are split before tau-cover;
- the greedy score is weighted by inverse full-row magnitude;
- tie-breaking is deterministic by exchange ID then original order;
- provider-linked and quantitative-reference exchanges are retained;
- compact priority metrics report `eta`, `eta_witness`, and `loss_max`.

## Intentional behaviour changes

- no GUI code or GUI-facing analysis objects;
- no PDF report generation in the clean package;
- regional CF ambiguity is always expanded explicitly and never resolved through a deterministic fallback mode;
- grouped-flow analyser warnings are always explicit.

## Unresolved scientific or technical uncertainties

- The clean package does not emit a full contribution ledger, so grouped-flow failures remain conservative screens only.
- The EcoSpold1 writer preserves structural validity, but it does not preserve original XML formatting or comments byte-for-byte.
- The rewrite is package-first but still lives inside the repository's shared root packaging metadata rather than a standalone `lca_flowkit` distribution root.
- The clean package currently omits some legacy convenience outputs such as the PDF report and GUI-facing analysis exports.
- Distinct CF candidates can now remain as separate scenario rows even when they lead to identical numeric contributions, provided their scientific metadata differs; this is intentional for auditability but can increase row counts.

## Validation status

Executed during this work:

- `PYTHONPYCACHEPREFIX=/private/tmp/lca_flowkit_pyc .venv/bin/python -m pytest tests/test_lca_flowkit_cf.py tests/test_lca_flowkit_analysis.py tests/test_lca_flowkit_workflows.py -q`
- `PYTHONPYCACHEPREFIX=/private/tmp/lca_flowkit_pyc .venv/bin/python -m pytest -q`

Result at completion:

- `194 passed`
- warnings were limited to existing `matplotlib`/`pyparsing` deprecation warnings from the environment.
