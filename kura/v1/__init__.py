"""
Kura V1: Procedural Implementation

A functional approach to conversation analysis that breaks down the pipeline
into composable functions for better flexibility and testability.
"""

from .kura import (
    # Core pipeline functions
    generate_base_clusters_from_conversation_summaries,
    reduce_clusters_from_base_clusters,
    reduce_dimensionality_from_clusters,
)
from kura.checkpoint import CheckpointManager
from kura.summarisation import summarise_conversations

__all__ = [
    # Core functions
    "summarise_conversations",
    "generate_base_clusters_from_conversation_summaries",
    "reduce_clusters_from_base_clusters",
    "reduce_dimensionality_from_clusters",
    # Utilities
    "CheckpointManager",
]

__version__ = "1.0.0"
