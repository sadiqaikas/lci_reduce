# AGENTS.md addition for `lca_flowkit`

## Purpose

Create a clean package implementation in `lca_flowkit/`.
The existing `lci_reduce/` implementation is the reference implementation. It may be read, tested against, and compared with, but it must not be modified.

The goal is not a GUI refactor. The goal is a smaller, clearer, publication-grade scientific Python package for LCIA-based database reduction and LCIA-critical flow priority-file generation.

## Design principles

* No GUI code.
* No desktop application logic.
* Package-first API design.
* Clear callable functions for scripts, notebooks, tests, and future CLI use.
* Modular code with small files. Keep modules below about 250 lines where practical.
* Small functions with explicit inputs and outputs.
* Comments should explain modelling decisions, assumptions, and edge cases.
* Avoid comments that merely describe obvious Python syntax.
* Prefer clarity over cleverness.
* Preserve scientifically valid behaviour from the existing implementation.
* If behaviour is changed, document why in `BEHAVIOUR_DIFFERENCES.md`.
* Do not silently drop unresolved data.
* Record all assumptions, warnings, and ambiguity decisions in metadata or diagnostics.

## Required public workflows

The package must provide three core workflows:

```python
reduce_database(...)
create_priority_file(...)
analyse_flows(...)
```

These functions should return structured Python objects or dictionaries. They must not only print text.

## Supported source formats

Support the source families already supported by the current project:

* openLCA JSON-LD databases and LCIA method archives;
* EcoSpold1 process exports;
* EcoSpold1 impact-method exports.

Parsing should be modular. Once parsed, internal objects should be format-neutral where practical.

## Scientific requirements

* Strict unit handling should be the default.
* Finite non-regional CF ambiguity must be represented explicitly.
* Regional CF ambiguity must be represented explicitly.
* Blank-location CFs are generic fallback factors.
* Same flow, same compartment, same unit, and many `Location` values is true regionalised CF structure.
* Different unit bases are unit-resolution cases, not regional ambiguity.
* Duplicate CFs may be collapsed only when value and relevant metadata are equivalent.
* Do not silently choose among ambiguous CFs.
* Regional CF ambiguity should be represented explicitly in scenario rows.
* Preserve positive and negative sign splitting where required.
* Explain tau-cover logic clearly in docstrings and comments.
* Single-flow eta is exact for single-flow omission.
* Grouped-flow eta is not generally equal to the sum of single-flow eta values.
* Grouped-flow analysis must report conservative bounds unless a full contribution ledger is used.

## Regional CF policy

The package should not expose a switch for turning regional expansion off.

If location-specific CFs are admissible for an exchange, represent that
ambiguity explicitly in scenario rows. If the represented scenario set grows
too large, raise `ScenarioExpansionError` rather than choosing one regional CF
implicitly.

## Testing expectations

Create controlled tests for:

* exact CF match;
* strict unit filtering;
* duplicate CF collapse;
* blank-location generic fallback;
* real regional CF ambiguity;
* scenario-expansion failure when regional rows exceed the configured limit;
* uncharacterised flows;
* malformed XML handling;
* grouped-flow analyser bounds.

Where useful, compare selected outputs against the existing `lci_reduce/` implementation. Any intentional difference must be documented.

## Deliverables

Inside `lca_flowkit/`, provide:

* package source code;
* tests;
* short README;
* minimal examples;
* `BEHAVIOUR_DIFFERENCES.md`;
* `IMPLEMENTATION_AUDIT.md`.

`IMPLEMENTATION_AUDIT.md` must summarise:

* package structure;
* public APIs;
* tests added;
* performance choices;
* behaviour copied from the old implementation;
* intentional behaviour changes;
* unresolved scientific or technical uncertainties.



## Documentation and comment standard

This package is scientific research software. Code readability is not enough. The implementation must also be scientifically auditable.

Comments and docstrings should explain modelling decisions, mathematical definitions, assumptions, and edge cases. It is acceptable for explanatory comments and docstrings to take substantial space. The soft 250-line module target applies mainly to executable code complexity, not to necessary scientific explanation.

For most public and internal scientific functions, include docstrings that explain:

* what scientific object the function represents;
* what assumptions it makes;
* what inputs are required;
* what output means;
* what ambiguity or fallback policy is applied;
* when the function is conservative rather than deterministic.

Where mathematics is used, include the formula in the docstring or nearby comments. Use plain text or simple LaTeX-style notation.

Examples:

```text
contribution = exchange_amount * characterisation_factor * unit_conversion
```

```text
eta_f = max shortfall caused by omitting one flow f from an already tau-covered row.
```

```text
Grouped eta is bounded conservatively because eta(f1 + f2) is not generally equal to eta(f1) + eta(f2).
```

Avoid comments that merely restate Python syntax. Prefer comments that explain why a decision is scientifically valid.
