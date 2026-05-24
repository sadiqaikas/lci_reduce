"""CSV manifest writing."""

from __future__ import annotations

import csv
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Iterable, Sequence


class ManifestWriter(AbstractContextManager["ManifestWriter"]):
    def __init__(self, path: Path, fieldnames: Sequence[str]) -> None:
        self.path = path
        self.fieldnames = list(fieldnames)
        self.handle = path.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.handle, fieldnames=self.fieldnames)
        self.writer.writeheader()

    def write_row(self, row: dict) -> None:
        self.writer.writerow({field: row.get(field, "") for field in self.fieldnames})

    def write_rows(self, rows: Iterable[dict]) -> None:
        for row in rows:
            self.write_row(row)

    def close(self) -> None:
        if not self.handle.closed:
            self.handle.close()

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        self.close()
        return None


def write_manifest_csv(path: Path, rows: Iterable[dict], fieldnames: Sequence[str]) -> None:
    with ManifestWriter(path, fieldnames) as writer:
        writer.write_rows(rows)
