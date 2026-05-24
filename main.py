"""
Convenience entrypoint.

Usage:
  python main.py           # start GUI
  python main.py gui       # start GUI
  python main.py cli ...   # run CLI (inspect/create)
  python -m lci_reduce     # start GUI from the package
"""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] == "gui":
        from lci_reduce.gui import main as gui_main

        return int(gui_main())
    if argv[0] == "cli":
        from lci_reduce.cli import main as cli_main

        return int(cli_main(argv[1:]))
    print("Usage: python main.py [gui|cli ...]", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
