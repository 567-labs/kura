"""
Procedural implementation of the Kura conversation analysis pipeline.

This module provides a functional approach to conversation analysis, breaking down
the pipeline into composable functions that can be used independently or together.

Key benefits over the class-based approach:
- Better composability and flexibility
- Easier testing of individual steps
- Clearer data flow and dependencies
- Better support for functional programming patterns
- Support for heterogeneous models through polymorphism
"""

import logging
from typing import Optional, TypeVar, List
from pydantic import BaseModel

# Import existing Kura components
from kura.base_classes import (
    BaseMetaClusterModel,
    BaseDimensionalityReduction,
    BaseCheckpointManager,
    BaseSummaryModel,
    BaseClusterDescriptionModel,
)
from kura.types import Conversation, ConversationSummary, Cluster
from kura.types.dimensionality import ProjectedCluster

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


async def summarise_conversations(
    conversations: List[Conversation],
    *,
    model: BaseSummaryModel,
    checkpoint_manager: Optional[BaseCheckpointManager] = None,
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
    model: BaseClusterDescriptionModel,
    checkpoint_manager: Optional[BaseCheckpointManager] = None,
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
    checkpoint_manager: Optional[BaseCheckpointManager] = None,
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
    checkpoint_manager: Optional[BaseCheckpointManager] = None,
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
