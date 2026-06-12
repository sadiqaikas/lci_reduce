"""Top-level input dispatch for database and LCIA method sources.

These helpers sit between the file-format readers and the public workflows.
They make two decisions explicit for audit purposes:

- which source family a path belongs to;
- which LCIA categories define the scientific scope of the run.
"""

from __future__ import annotations

from .ecospold1 import load_ecospold1_bundle
from .errors import DataFormatError
from .jsonld import load_jsonld_bundle
from .method_selection import select_categories
from .models import DatasetBundle, ImpactCategory
from .utils import detect_source_family, iter_source_entries


def load_bundle(source: str) -> DatasetBundle:
    """Load one source bundle by detecting whether it is JSON-LD or EcoSpold1.

    The function deliberately delegates the actual parsing to a format-specific
    reader after the family has been identified.  That keeps downstream code
    free from ``if jsonld else ecospold`` branches.
    """
    entries = iter_source_entries(source)
    family = detect_source_family(entries)
    if family == "jsonld":
        return load_jsonld_bundle(source, entries)
    if family == "ecospold1":
        return load_ecospold1_bundle(source, entries)
    raise DataFormatError(f"Unsupported detected family: {family}")


def resolve_categories(database_bundle: DatasetBundle, methods_path: str | None, method_selection: str) -> list[ImpactCategory]:
    """Merge optional external methods and apply the caller's category filter.

    Downstream workflows only work with one resolved list of impact categories.
    Keeping the merge here makes the scientific scope of a run explicit.

    If an external methods source is supplied, it must match the database input
    family.  Cross-family matching would create a scientifically unclear basis
    for CF resolution, so the package rejects it.
    """
    categories: dict[str, ImpactCategory] = dict(database_bundle.categories)
    if methods_path:
        methods_bundle = load_bundle(methods_path)
        if methods_bundle.source_format != database_bundle.source_format:
            raise DataFormatError(
                f"Database source format {database_bundle.source_format} is not compatible with methods source format {methods_bundle.source_format}"
            )
        categories.update(methods_bundle.categories)
        parse_warnings = methods_bundle.extra.get("parse_warnings", [])
        if parse_warnings:
            database_bundle.extra.setdefault("parse_warnings", []).extend(parse_warnings)
    selected = select_categories(list(categories.values()), method_selection)
    if not selected:
        raise DataFormatError("No usable LCIA categories were available for this run")
    return selected
