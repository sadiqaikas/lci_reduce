"""Custom errors for lci_reduce."""


class LciReduceError(Exception):
    """Base application error."""


class CoverageError(LciReduceError):
    """Raised when tau coverage cannot be satisfied."""


class DataFormatError(LciReduceError):
    """Raised when JSON-LD input structure is unsupported."""


class NativeArchiveConversionError(LciReduceError):
    """Raised when a native openLCA archive cannot be converted to JSON-LD."""


class CharacterisationFactorError(LciReduceError):
    """Base error for characterisation factor resolution failures."""


class UnitCompatibilityError(CharacterisationFactorError):
    """Raised when exchange units and characterisation factor units mismatch."""


class AmbiguousCharacterisationFactorError(CharacterisationFactorError):
    """Raised when multiple CF candidates remain after strict disambiguation."""


class DuplicateMethodConflictError(CharacterisationFactorError):
    """Raised when duplicate LCIA method/category UUIDs conflict."""


class AmbiguousMappingError(AmbiguousCharacterisationFactorError):
    """Backward-compatible alias for ambiguous CF mapping failures."""


class MissingFlowError(LciReduceError):
    """Raised when an exchange flow reference cannot be resolved."""


class UncharacterisedExchangeError(LciReduceError):
    """Raised when uncharacterised-policy=fail is violated."""


class RunCancelledError(LciReduceError):
    """Raised when the GUI user cancels a running operation."""
