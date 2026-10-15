"""vsm_mini: a minimal, dependency-free prototype of Versioned Semantic Memory."""
from .baselines import MutableStore, NaiveRAG
from .engine import (Event, HistoryUnavailable, Layer, LayeredStore, NotSupported, Op,
                     Record, RollbackError)

__all__ = ["Event", "Op", "Record", "Layer", "LayeredStore", "NaiveRAG", "MutableStore",
           "NotSupported", "RollbackError", "HistoryUnavailable"]
__version__ = "0.1.0"
