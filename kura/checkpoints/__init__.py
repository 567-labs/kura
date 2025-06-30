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

import importlib.util

from kura.base_classes import BaseCheckpointManager
from .jsonl import JSONLCheckpointManager
from .multi import MultiCheckpointManager

# Check if optional dependencies are available
PARQUET_AVAILABLE = importlib.util.find_spec("pyarrow") is not None
HF_DATASETS_AVAILABLE = importlib.util.find_spec("datasets") is not None

# Base exports
__all__ = [
    "BaseCheckpointManager",
    "JSONLCheckpointManager",
    "MultiCheckpointManager",
    "PARQUET_AVAILABLE",
    "HF_DATASETS_AVAILABLE",
]

# Conditional imports and exports
if PARQUET_AVAILABLE:
    from .parquet import ParquetCheckpointManager as ParquetCheckpointManager
    __all__.append("ParquetCheckpointManager")

if HF_DATASETS_AVAILABLE:
    from .hf_dataset import HFDatasetCheckpointManager as HFDatasetCheckpointManager
    __all__.append("HFDatasetCheckpointManager")
