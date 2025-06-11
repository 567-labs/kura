"""
Checkpoint management system for Kura.

This module provides different checkpoint backends for storing and loading
intermediate pipeline results. The available backends are:

- BaseCheckpointManager: Abstract base class for all checkpoint managers
- JSONLCheckpointManager: Traditional JSONL file-based checkpoints (default)
- ParquetCheckpointManager: Parquet-based checkpoints for better compression
- HFDatasetCheckpointManager: HuggingFace datasets-based checkpoints
- MultiCheckpointManager: Coordinate multiple checkpoint backends

The ParquetCheckpointManager provides better compression (50% space savings)
and faster loading for analytical workloads, while HFDatasetCheckpointManager
provides advanced features like streaming, versioning, and cloud storage integration.

The MultiCheckpointManager allows using multiple backends simultaneously for
redundancy and performance optimization.
"""

from kura.base_classes import BaseCheckpointManager
from .jsonl import JSONLCheckpointManager
from .multi import MultiCheckpointManager

# Import ParquetCheckpointManager if PyArrow is available
try:
    from .parquet import ParquetCheckpointManager

    PARQUET_AVAILABLE = True
except ImportError:
    ParquetCheckpointManager = None
    PARQUET_AVAILABLE = False

# Import HFDatasetCheckpointManager if datasets is available
try:
    from .hf_dataset import HFDatasetCheckpointManager, HF_DATASETS_AVAILABLE
except ImportError:
    HFDatasetCheckpointManager = None
    HF_DATASETS_AVAILABLE = False

__all__ = [
    "BaseCheckpointManager",
    "JSONLCheckpointManager",
    "MultiCheckpointManager",
    "PARQUET_AVAILABLE",
    "HF_DATASETS_AVAILABLE",
]

# Add ParquetCheckpointManager to exports if available
if PARQUET_AVAILABLE:
    __all__.append("ParquetCheckpointManager")

if HF_DATASETS_AVAILABLE:
    __all__.append("HFDatasetCheckpointManager")
