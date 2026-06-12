# lca_flowkit

`lca_flowkit` is a package-first rewrite of the LCIA-conditioned reduction workflows in this repository.

It is intentionally separate from the legacy `lci_reduce` package:

- no GUI code;
- no desktop application state;
- no openLCA IPC;
- JSON-LD ZIP and EcoSpold1 inputs only;
- explicit Python APIs for scripts, notebooks, and tests.

## Public APIs

```python
from lca_flowkit import analyse_flows, create_priority_file, reduce_database
```

### Reduce a database

```python
result = reduce_database(
    database_path="database.zip",
    methods_path=None,
    output_dir="out/reduced",
    method_selection="all",
    tau=0.95,
    strict_units=True,
)
```

Key outputs:

- reduced database ZIP;
- `reduction_metadata.json`;
- `process_manifest.csv`;
- `exchange_manifest.csv`;
- `validation.json`;
- `warnings.csv`.

### Create a compact LCIA-critical priority file

```python
result = create_priority_file(
    database_path="database.zip",
    methods_path=None,
    output_dir="out/priority",
    method_selection="all",
    audit_tau=[0.95, 0.99],
    strict_units=True,
)
```

Key outputs:

- `lcia_flow_priority.csv`;
- `lcia_flow_priority_metadata.json`;
- `warnings.csv`.

### Analyse selected flows from a priority file

```python
report = analyse_flows(
    priority_file="out/priority/lcia_flow_priority.csv",
    metadata_file="out/priority/lcia_flow_priority_metadata.json",
    flow_ids=["flow-uuid-1"],
    flow_names=["Carbon dioxide, fossil"],
    result_types=["eta_bounds", "loss_max_flows", "coverage_summary"],
    tau=0.95,
    top_n=20,
)
```

To analyse the entire compact priority file:

```python
report = analyse_flows(
    priority_file="out/priority/lcia_flow_priority.csv",
    metadata_file="out/priority/lcia_flow_priority_metadata.json",
    all_flows=True,
    tau=0.95,
)
```

The analyser returns structured Python data, including:

- matched and unmatched requests;
- ambiguous flow-name matches;
- exact single-flow `eta` values;
- conservative grouped-flow `eta` bounds;
- witness process/category/sign strings;
- top and least repair rankings;
- compact warnings about grouped-flow interpretation.

## Scientific notes

- Reduction uses characterised contributions, not raw exchange amounts.
- Positive and negative contributions are split before greedy selection.
- Finite and regional CF ambiguity are both represented explicitly in scenario rows.
- If regional scenario expansion becomes too large, the workflows raise `ScenarioExpansionError` instead of choosing one regional CF.
- Greedy selection uses weighted marginal fill:
  `score[e] = sum_j min(contribution[j, e], remaining_demand[j]) / full[j]`
- Single-flow `eta` is exact for single-flow omission only.
- Grouped-flow `eta` from the compact CSV is conservative, not exact.
- `result_types` filters which analytical sections `analyse_flows` returns, while match bookkeeping is always included.
- `all_flows=True` analyses every compact-priority row; it must not be combined with explicit `flow_ids` or `flow_names`.

## Scope

Implemented here:

- JSON-LD reduction;
- EcoSpold1 reduction;
- JSON-LD priority-file generation;
- EcoSpold1 priority-file generation;
- package-style compact priority analysis.

Not implemented here:

- GUI workflows;
- PDF report generation;
- openLCA IPC;
- whole-database exact optimisation;
- grouped-flow exact failure analysis without a ledger.
