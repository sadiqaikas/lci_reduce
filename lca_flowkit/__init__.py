"""Public package surface for the `lca_flowkit` scientific workflows.

The package intentionally exposes a small API:

- `reduce_database` for LCIA-conditioned database reduction;
- `create_priority_file` for compact LCIA-critical flow sidecars;
- `analyse_flows` for notebook and script analysis of those sidecars.
"""

from .analysis import analyse_flows
from .priority import create_priority_file
from .reducer import reduce_database

__all__ = [
    "analyse_flows",
    "create_priority_file",
    "reduce_database",
]
