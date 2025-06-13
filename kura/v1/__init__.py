"""
Kura V1: Procedural Implementation

A functional approach to conversation analysis that breaks down the pipeline
into composable functions for better flexibility and testability.
"""

from .kura import (
    # Core pipeline functions
    reduce_clusters_from_base_clusters,
    reduce_dimensionality_from_clusters,
)

# Import ParquetCheckpointManager if pyarrow is available
try:
    from kura.checkpoints.parquet import ParquetCheckpointManager

    PARQUET_AVAILABLE = True
except ImportError:
    ParquetCheckpointManager = None
    PARQUET_AVAILABLE = False

try:
    from kura.checkpoints.hf_dataset import HFDatasetCheckpointManager

    HF_AVAILABLE = True
except ImportError:
    HFDatasetCheckpointManager = None
    HF_AVAILABLE = False

__all__ = [
    "reduce_clusters_from_base_clusters",
    "reduce_dimensionality_from_clusters",
]

# Add ParquetCheckpointManager to __all__ if available
if PARQUET_AVAILABLE:
    __all__.append("ParquetCheckpointManager")

if HF_AVAILABLE:
    __all__.append("HFDatasetCheckpointManager")

__version__ = "1.0.0"
