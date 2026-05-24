"""Package entrypoint for `python -m lci_reduce`."""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] == "gui":
        from .gui import main as gui_main

        return int(gui_main())
    if argv[0] == "cli":
        from .cli import main as cli_main

        return int(cli_main(argv[1:]))
    print("Usage: python -m lci_reduce [gui|cli ...]", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
