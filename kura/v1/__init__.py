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

__all__ = [
    "reduce_clusters_from_base_clusters",
    "reduce_dimensionality_from_clusters",
]

__version__ = "1.0.0"
