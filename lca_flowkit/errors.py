"""Package-local exceptions."""

from __future__ import annotations


class CleanLciReduceError(Exception):
    """Base class for `lca_flowkit` workflow failures."""


class DataFormatError(CleanLciReduceError):
    """Raised when an input archive cannot be parsed safely."""


class UnitResolutionError(CleanLciReduceError):
    """Raised when strict unit handling cannot confirm compatibility."""


class CoverageError(CleanLciReduceError):
    """Raised when a tau certificate cannot be satisfied or verified."""


class ScenarioExpansionError(CleanLciReduceError):
    """Raised when ambiguity expansion exceeds configured limits."""


class AnalysisError(CleanLciReduceError):
    """Raised when a compact priority file cannot be analysed safely."""
