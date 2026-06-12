"""Helpers for selecting LCIA categories from a loaded methods bundle.

Method selection is intentionally simple and string-based.  The goal is not to
invent a complex query language, but to make the reduction scope easy to state
and inspect in metadata.
"""

from __future__ import annotations

from .models import ImpactCategory
from .utils import fold_text


def parse_method_selection(selection: str) -> tuple[str, str]:
    """Parse the compact method-selection syntax used by the public APIs.

    Supported forms:

    - ``all``
    - ``family:<text>``
    - ``method:<text>``
    - ``category:<text>``
    """
    token = (selection or "all").strip()
    if token == "all":
        return "all", ""
    if ":" not in token:
        raise ValueError("method_selection must be all, family:<text>, method:<text>, or category:<text>")
    mode, query = token.split(":", 1)
    mode = mode.strip().lower()
    query = query.strip()
    if mode not in {"family", "method", "category"} or not query:
        raise ValueError("invalid method_selection")
    return mode, query


def select_categories(categories: list[ImpactCategory], selection: str) -> list[ImpactCategory]:
    """Filter categories by family, method, category name, or ``all``.

    Matching is substring-based after text normalisation.  That keeps the API
    forgiving enough for notebook use while staying deterministic.
    """
    mode, query = parse_method_selection(selection)
    if mode == "all":
        return list(categories)
    needle = fold_text(query)
    selected: list[ImpactCategory] = []
    for category in categories:
        family_text = fold_text(category.path)
        method_text = fold_text(category.method_name)
        category_text = fold_text(category.name)
        if mode == "family" and needle in family_text:
            selected.append(category)
        elif mode == "method" and needle in method_text:
            selected.append(category)
        elif mode == "category" and needle in category_text:
            selected.append(category)
    return selected
