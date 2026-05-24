# lci_reduce

`lci_reduce` is a ZIP-only Python application that creates a reduced openLCA JSON-LD database ZIP from an original JSON-LD database archive.

It does not use openLCA IPC, does not connect to openLCA, and does not edit databases in place.

## Workflow

Input:

- original openLCA JSON-LD database archive (`.zip` or `.zolca`)
- optional LCIA methods JSON-LD archive (`.zip` or `.zolca`) or folder
- tau
- LCIA indicator selection

Output:

- lite JSON-LD database ZIP
- `run_summary.json`
- `exchange_manifest.csv`
- `process_manifest.csv`
- `validation.json`
- `config.json`
- `warnings.csv`
- short PDF report

Additional sidecar workflow:

- `lcia_flow_priority.csv`
- `lcia_flow_priority_metadata.json`

Priority analyser workflow:

- input: existing `lcia_flow_priority.csv`
- optional input: `lcia_flow_priority_metadata.json`
- no original database ZIP required
- no failed mappings CSV required
- no lite database ZIP generated

## Reduction rule

For each process and each selected LCIA category, the tool computes characterised contributions:

`A[j, e] = exchange_amount(e) * CF[j, flow(e)]`

The tool splits signed contributions:

- `A_pos = max(A, 0)`
- `A_neg = max(-A, 0)`

It then runs the deterministic greedy tau-cover independently on `A_pos` and `A_neg`, retaining:

`selected = selected_pos OR selected_neg OR protected`

Only unselected elementary exchange objects may be removed from process exchange lists. Non-elementary exchanges, provider-linked exchanges, quantitative reference exchanges, and protected uncharacterised elementary exchanges are preserved.

## CLI

Inspect:

```bash
lci_reduce inspect \
  --database original_database.zip \
  --methods methods.zip
```

If the database archive already contains LCIA methods and categories, `inspect` reports that and you can leave `--methods` empty.
The loader accepts JSON-LD folders, JSON-LD `.zip` archives, JSON-LD `.zolca` wrappers, and native openLCA `.zolca` / Derby backup archives. Native openLCA archives are converted offline and read-only to temporary JSON-LD before reduction. SimaPro `.sip` packages are still not supported directly.

Create:

```bash
lci_reduce create \
  --database original_database.zip \
  --methods methods.zip \
  --output out_folder \
  --tau 0.95 \
  --method-selection all \
  --uncharacterised-policy keep \
  --strict-units true
```

Flow priority:

```bash
lci_reduce priority \
  --database original_database.zip \
  --methods methods.zip \
  --output out_folder \
  --method-selection all \
  --audit-tau 0.95 0.99 \
  --strict-units true
```

Priority analyser:

```bash
lci_reduce analyse-priority \
  --priority-csv lcia_flow_priority.csv \
  --metadata-json lcia_flow_priority_metadata.json \
  --audit-tau 0.95 \
  --top-n 20 \
  --select-flow-id flow-1,flow-2 \
  --select-flow-name "Sulfur dioxide" \
  --select-flow-name "Methane, fossil" \
  --output-ranked-csv ranked.csv \
  --output-summary-json priority_analysis_summary.json
```

If a flow name contains commas, prefer repeating `--select-flow-name` instead of placing several names into one comma-separated argument.

## Priority analyser mathematics

The analyser uses the compact priority CSV only.

- `eta_tau` is the exact single-flow certificate shortfall for omitting that one flow at the chosen audit tau.
- `loss_max_tau` is the maximum raw retained-coverage loss before overshoot margin.
- For a selected flow set `F`, the analyser does not claim exact combined `eta_F`.
- Instead it reports the compact-screen bound:

`max_{f in F} eta_f(tau) <= exact group eta_F(tau) <= min(tau, sum_{f in F} loss_max_f(tau))`

Interpret this as a conservative screen:

- low upper bounds suggest low compact-screen group risk
- positive lower bounds mean at least one selected flow already breaks the certificate on its own
- large upper bounds with zero lower bound mean accumulation risk may still matter

Exact grouped failed-set audit would require a future ledger or recomputation mode.

## GUI

The GUI uses the same backend as the CLI and now includes:

- a reduction tab for inspection and run execution
- a flow-priority tab for LCIA transfer-priority sidecars
- a priority-analyser tab for compact priority CSV screening
- a reduction-curves tab for comparing completed runs across multiple tau values
- a CLI info tab with copy-ready commands and workflow guidance

```bash
lci_reduce-gui
```

When you select a database archive, the GUI checks whether the database already contains impact methods and shows a simple hint. If it does, you can use the `Use Database Methods` button and leave the optional methods input empty.

Shortcut (no console script needed):

```bash
python main.py
```

Package entrypoint:

```bash
python -m lci_reduce
```

## Development

Install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install ".[dev]"
```

If you want an editable install for development, upgrade `pip` first and then use:

```bash
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

Run tests:

```bash
pytest
```

## Notes

- The tool is ZIP-only.
- The input archive is never modified.
- IDs and UUIDs are preserved.
- Unused flow objects may remain in the output.
- Unit handling is strict by default. Ambiguous or incompatible CF unit usage fails in strict mode.
- `priority` does not create a lite database ZIP.
- `analyse-priority` does not require the original database and does not rewrite any database.
