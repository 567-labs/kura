"""
Procedural implementation of the Kura conversation analysis pipeline.

This module provides a functional approach to conversation analysis, breaking down
the pipeline into composable functions that can be used independently or together.

Key benefits of this design:
- Better composability and flexibility
- Easier testing of individual steps
- Clearer data flow and dependencies
- Better support for functional programming patterns
- Support for heterogeneous models through polymorphism
"""

import logging
from typing import Optional, TypeVar, List
import os
from pydantic import BaseModel

# Import existing Kura components
from kura.base_classes import (
    BaseSummaryModel,
    BaseClusterModel,
    BaseMetaClusterModel,
    BaseDimensionalityReduction,
)
from kura.types import Conversation, Cluster, ConversationSummary
from kura.types.dimensionality import ProjectedCluster

# Set up logger
logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


# =============================================================================
# Checkpoint Management
# =============================================================================


class CheckpointManager:
    """Handles checkpoint loading and saving for pipeline steps."""

    def __init__(self, checkpoint_dir: str, *, enabled: bool = True):
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory for saving checkpoints
            enabled: Whether checkpointing is enabled
        """
        self.checkpoint_dir = checkpoint_dir
        self.enabled = enabled

        if self.enabled:
            self.setup_checkpoint_dir()

    def setup_checkpoint_dir(self) -> None:
        """Create checkpoint directory if it doesn't exist."""
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
            logger.info(f"Created checkpoint directory: {self.checkpoint_dir}")

    def get_checkpoint_path(self, filename: str) -> str:
        """Get full path for a checkpoint file."""
        return os.path.join(self.checkpoint_dir, filename)

    def load_checkpoint(self, filename: str, model_class: type[T]) -> Optional[List[T]]:
        """Load data from a checkpoint file if it exists.

        Args:
            filename: Name of the checkpoint file
            model_class: Pydantic model class for deserializing the data

        Returns:
            List of model instances if checkpoint exists, None otherwise
        """
        if not self.enabled:
            return None

        checkpoint_path = self.get_checkpoint_path(filename)
        if os.path.exists(checkpoint_path):
            logger.info(
                f"Loading checkpoint from {checkpoint_path} for {model_class.__name__}"
            )
            with open(checkpoint_path, "r") as f:
                return [model_class.model_validate_json(line) for line in f]
        return None

    def save_checkpoint(self, filename: str, data: List[T]) -> None:
        """Save data to a checkpoint file.

        Args:
            filename: Name of the checkpoint file
            data: List of model instances to save
        """
        if not self.enabled:
            return

        checkpoint_path = self.get_checkpoint_path(filename)
        with open(checkpoint_path, "w") as f:
            for item in data:
                f.write(item.model_dump_json() + "\n")
        logger.info(f"Saved checkpoint to {checkpoint_path} with {len(data)} items")


# =============================================================================
# Core Pipeline Functions
# =============================================================================


async def summarise_conversations(
    conversations: List[Conversation],
    *,
    model: BaseSummaryModel,
    checkpoint_manager: Optional[CheckpointManager] = None,
) -> List[ConversationSummary]:
    """Generate summaries for a list of conversations.

    This is a pure function that takes conversations and a summary model,
    and returns conversation summaries. Optionally uses checkpointing.

    The function works with any model that implements BaseSummaryModel,
    supporting heterogeneous backends (OpenAI, vLLM, Hugging Face, etc.)
    through polymorphism.

    Args:
        conversations: List of conversations to summarize
        model: Model to use for summarization (OpenAI, vLLM, local, etc.)
        checkpoint_manager: Optional checkpoint manager for caching

    Returns:
        List of conversation summaries

    Example:
        >>> openai_model = OpenAISummaryModel(api_key="sk-...")
        >>> checkpoint_mgr = CheckpointManager("./checkpoints")
        >>> summaries = await summarise_conversations(
        ...     conversations=my_conversations,
        ...     model=openai_model,
        ...     checkpoint_manager=checkpoint_mgr
        ... )
    """
    logger.info(
        f"Starting summarization of {len(conversations)} conversations using {type(model).__name__}"
    )

    # Try to load from checkpoint
    if checkpoint_manager:
        cached = checkpoint_manager.load_checkpoint(
            model.checkpoint_filename, ConversationSummary
        )
        if cached:
            logger.info(f"Loaded {len(cached)} summaries from checkpoint")
            return cached

    # Generate summaries
    logger.info("Generating new summaries...")
    summaries = await model.summarise(conversations)
    logger.info(f"Generated {len(summaries)} summaries")

    # Save to checkpoint
    if checkpoint_manager:
        logger.info(f"Saving summaries to checkpoint: {model.checkpoint_filename}")
        checkpoint_manager.save_checkpoint(model.checkpoint_filename, summaries)

    return summaries


async def generate_base_clusters_from_conversation_summaries(
    summaries: List[ConversationSummary],
    *,
    model: BaseClusterModel,
    checkpoint_manager: Optional[CheckpointManager] = None,
) -> List[Cluster]:
    """Generate base clusters from conversation summaries.

    This function groups similar summaries into initial clusters using
    the provided clustering model. Supports different clustering algorithms
    through the model interface.

    Args:
        summaries: List of conversation summaries to cluster
        model: Model to use for clustering (HDBSCAN, KMeans, etc.)
        checkpoint_manager: Optional checkpoint manager for caching

    Returns:
        List of base clusters

    Example:
        >>> cluster_model = ClusterModel(algorithm="hdbscan")
        >>> clusters = await generate_base_clusters(
        ...     summaries=conversation_summaries,
        ...     model=cluster_model,
        ...     checkpoint_manager=checkpoint_mgr
        ... )
    """
    logger.info(
        f"Starting clustering of {len(summaries)} summaries using {type(model).__name__}"
    )

    # Try to load from checkpoint
    if checkpoint_manager:
        cached = checkpoint_manager.load_checkpoint(model.checkpoint_filename, Cluster)
        if cached:
            logger.info(f"Loaded {len(cached)} clusters from checkpoint")
            return cached

    # Generate clusters
    logger.info("Generating new clusters...")
    clusters = await model.cluster_summaries(summaries)
    logger.info(f"Generated {len(clusters)} clusters")

    # Save to checkpoint
    if checkpoint_manager:
        checkpoint_manager.save_checkpoint(model.checkpoint_filename, clusters)

    return clusters


async def reduce_clusters_from_base_clusters(
    clusters: List[Cluster],
    *,
    model: BaseMetaClusterModel,
    checkpoint_manager: Optional[CheckpointManager] = None,
) -> List[Cluster]:
    """Reduce clusters into a hierarchical structure.

    Iteratively combines similar clusters until the number of root clusters
    is less than or equal to the model's max_clusters setting.

    Args:
        clusters: List of initial clusters to reduce
        model: Meta-clustering model to use for reduction
        checkpoint_manager: Optional checkpoint manager for caching

    Returns:
        List of clusters with hierarchical structure

    Example:
        >>> meta_model = MetaClusterModel(max_clusters=5)
        >>> reduced = await reduce_clusters(
        ...     clusters=base_clusters,
        ...     model=meta_model,
        ...     checkpoint_manager=checkpoint_mgr
        ... )
    """
    logger.info(
        f"Starting cluster reduction from {len(clusters)} initial clusters using {type(model).__name__}"
    )

    # Try to load from checkpoint
    if checkpoint_manager:
        cached = checkpoint_manager.load_checkpoint(model.checkpoint_filename, Cluster)
        if cached:
            root_count = len([c for c in cached if c.parent_id is None])
            logger.info(
                f"Loaded {len(cached)} clusters from checkpoint ({root_count} root clusters)"
            )
            return cached

    # Start with all clusters as potential roots
    all_clusters = clusters.copy()
    root_clusters = clusters.copy()

    # Get max_clusters from model if available, otherwise use default
    max_clusters = getattr(model, "max_clusters", 10)
    logger.info(f"Starting with {len(root_clusters)} clusters, target: {max_clusters}")

    # Iteratively reduce until we have desired number of root clusters
    while len(root_clusters) > max_clusters:
        # Get updated clusters from meta-clustering
        new_current_level = await model.reduce_clusters(root_clusters)

        # Find new root clusters (those without parents)
        root_clusters = [c for c in new_current_level if c.parent_id is None]

        # Remove old clusters that now have parents
        old_cluster_ids = {c.id for c in new_current_level if c.parent_id}
        all_clusters = [c for c in all_clusters if c.id not in old_cluster_ids]

        # Add new clusters to the complete list
        all_clusters.extend(new_current_level)

        logger.info(f"Reduced to {len(root_clusters)} root clusters")

    logger.info(
        f"Cluster reduction complete: {len(all_clusters)} total clusters, {len(root_clusters)} root clusters"
    )

    # Save to checkpoint
    if checkpoint_manager:
        checkpoint_manager.save_checkpoint(model.checkpoint_filename, all_clusters)

    return all_clusters


async def reduce_dimensionality_from_clusters(
    clusters: List[Cluster],
    *,
    model: BaseDimensionalityReduction,
    checkpoint_manager: Optional[CheckpointManager] = None,
) -> List[ProjectedCluster]:
    """Reduce dimensions of clusters for visualization.

    Projects clusters to 2D space using the provided dimensionality reduction model.
    Supports different algorithms (UMAP, t-SNE, PCA, etc.) through the model interface.

    Args:
        clusters: List of clusters to project
        model: Dimensionality reduction model to use (UMAP, t-SNE, etc.)
        checkpoint_manager: Optional checkpoint manager for caching

    Returns:
        List of projected clusters with 2D coordinates

    Example:
        >>> dim_model = HDBUMAP(n_components=2)
        >>> projected = await reduce_dimensionality(
        ...     clusters=hierarchical_clusters,
        ...     model=dim_model,
        ...     checkpoint_manager=checkpoint_mgr
        ... )
    """
    logger.info(
        f"Starting dimensionality reduction for {len(clusters)} clusters using {type(model).__name__}"
    )

    # Try to load from checkpoint
    if checkpoint_manager:
        cached = checkpoint_manager.load_checkpoint(
            model.checkpoint_filename, ProjectedCluster
        )
        if cached:
            logger.info(f"Loaded {len(cached)} projected clusters from checkpoint")
            return cached

    # Reduce dimensionality
    logger.info("Projecting clusters to 2D space...")
    projected_clusters = await model.reduce_dimensionality(clusters)
    logger.info(f"Projected {len(projected_clusters)} clusters to 2D")

    # Save to checkpoint
    if checkpoint_manager:
        checkpoint_manager.save_checkpoint(
            model.checkpoint_filename, projected_clusters
        )

    return projected_clusters
