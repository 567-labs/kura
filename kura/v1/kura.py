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
from typing import Optional, TypeVar, List, Literal
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


class MultiCheckpointManager(BaseCheckpointManager):
    """Manages multiple checkpoint managers for redundancy, performance, or separation.

    This class coordinates multiple checkpoint managers, allowing you to:
    - Save to multiple backends for redundancy (e.g., local + cloud backup)
    - Load from the fastest available source
    - Separate environments (dev/staging/prod)
    - Implement different retention policies

    Attributes:
        managers: List of checkpoint managers to coordinate
        save_strategy: How to save checkpoints:
            - "all_enabled": Save to all enabled managers (default)
            - "primary_only": Save only to the first enabled manager
        load_strategy: How to load checkpoints:
            - "first_found": Return first successful load (default)
            - "priority": Try managers in order, return first success

    Example:
        >>> local_mgr = CheckpointManager("./local_checkpoints")
        >>> cloud_mgr = CheckpointManager("./cloud_backup")
        >>> multi_mgr = MultiCheckpointManager(
        ...     managers=[local_mgr, cloud_mgr],
        ...     save_strategy="all_enabled",
        ...     load_strategy="first_found"
        ... )
    """

    def __init__(
        self,
        managers: List[BaseCheckpointManager],
        *,
        save_strategy: Literal["all_enabled", "primary_only"] = "all_enabled",
        load_strategy: Literal["first_found", "priority"] = "first_found",
    ):
        """Initialize multi-checkpoint manager.

        Args:
            managers: List of checkpoint managers to coordinate
            save_strategy: Strategy for saving checkpoints
            load_strategy: Strategy for loading checkpoints
        """
        if not managers:
            raise ValueError("At least one checkpoint manager must be provided")

        self.managers = managers
        self.save_strategy = save_strategy
        self.load_strategy = load_strategy

        # Consider enabled if any manager is enabled
        enabled = any(mgr.enabled for mgr in managers)

        # Initialize base class with dummy checkpoint_dir
        super().__init__("", enabled=enabled)

        logger.info(
            f"Initialized MultiCheckpointManager with {len(managers)} managers "
            f"(save: {save_strategy}, load: {load_strategy})"
        )

    def setup_checkpoint_dir(self) -> None:
        """Setup checkpoint directories for all managers."""
        for mgr in self.managers:
            if mgr.enabled:
                mgr.setup_checkpoint_dir()

    def load_checkpoint(
        self, filename: str, model_class: type[T], **kwargs
    ) -> Optional[List[T]]:
        """Load checkpoint from first available manager.

        Args:
            filename: Name of checkpoint file
            model_class: Pydantic model class for deserializing
            **kwargs: Additional arguments passed to managers

        Returns:
            Loaded data if found, None otherwise
        """
        if not self.enabled:
            return None

        if self.load_strategy == "first_found":
            # Try all managers, return first successful load
            for mgr in self.managers:
                if mgr.enabled:
                    try:
                        data = mgr.load_checkpoint(filename, model_class, **kwargs)
                        if data is not None:
                            logger.info(
                                f"Loaded {filename} from {mgr.__class__.__name__}"
                            )
                            return data
                    except Exception as e:
                        logger.warning(
                            f"Failed to load {filename} from {mgr.__class__.__name__}: {e}"
                        )
                        continue

        elif self.load_strategy == "priority":
            # Try managers in order, return first success
            for mgr in self.managers:
                if mgr.enabled:
                    try:
                        data = mgr.load_checkpoint(filename, model_class, **kwargs)
                        if data is not None:
                            logger.info(
                                f"Loaded {filename} from {mgr.__class__.__name__} (priority)"
                            )
                            return data
                    except Exception as e:
                        logger.warning(
                            f"Failed to load {filename} from {mgr.__class__.__name__}: {e}"
                        )
                        # In priority mode, don't continue to next manager on error
                        break

        logger.info(f"No checkpoint found for {filename} in any manager")
        return None

    def save_checkpoint(self, filename: str, data: List[T], **kwargs) -> None:
        """Save checkpoint according to configured strategy.

        Args:
            filename: Name of checkpoint file
            data: Data to save
            **kwargs: Additional arguments passed to managers
        """
        if not self.enabled:
            return

        if self.save_strategy == "all_enabled":
            # Save to all enabled managers
            saved_count = 0
            for mgr in self.managers:
                if mgr.enabled:
                    try:
                        mgr.save_checkpoint(filename, data, **kwargs)
                        saved_count += 1
                        logger.debug(f"Saved {filename} to {mgr.__class__.__name__}")
                    except Exception as e:
                        logger.warning(
                            f"Failed to save {filename} to {mgr.__class__.__name__}: {e}"
                        )

            if saved_count > 0:
                logger.info(f"Saved {filename} to {saved_count} checkpoint manager(s)")
            else:
                logger.error(f"Failed to save {filename} to any checkpoint manager")

        elif self.save_strategy == "primary_only":
            # Save only to first enabled manager
            for mgr in self.managers:
                if mgr.enabled:
                    try:
                        mgr.save_checkpoint(filename, data, **kwargs)
                        logger.info(
                            f"Saved {filename} to primary manager {mgr.__class__.__name__}"
                        )
                        return
                    except Exception as e:
                        logger.error(
                            f"Failed to save {filename} to primary manager: {e}"
                        )
                        raise

            logger.warning("No enabled checkpoint managers available for saving")

    def list_checkpoints(self) -> List[str]:
        """List all unique checkpoints across all managers.

        Returns:
            Combined list of unique checkpoint filenames
        """
        all_checkpoints = set()

        for mgr in self.managers:
            if mgr.enabled:
                try:
                    checkpoints = mgr.list_checkpoints()
                    all_checkpoints.update(checkpoints)
                except Exception as e:
                    logger.warning(
                        f"Failed to list checkpoints from {mgr.__class__.__name__}: {e}"
                    )

        return sorted(list(all_checkpoints))

    def delete_checkpoint(self, filename: str) -> bool:
        """Delete checkpoint from all managers.

        Args:
            filename: Name of checkpoint file to delete

        Returns:
            True if deleted from at least one manager
        """
        if not self.enabled:
            return False

        deleted = False
        for mgr in self.managers:
            if mgr.enabled:
                try:
                    if mgr.delete_checkpoint(filename):
                        deleted = True
                        logger.debug(
                            f"Deleted {filename} from {mgr.__class__.__name__}"
                        )
                except Exception as e:
                    logger.warning(
                        f"Failed to delete {filename} from {mgr.__class__.__name__}: {e}"
                    )

        if deleted:
            logger.info(f"Deleted {filename} from checkpoint manager(s)")

        return deleted

    def __repr__(self) -> str:
        """String representation of multi-checkpoint manager."""
        manager_names = [mgr.__class__.__name__ for mgr in self.managers]
        return (
            f"MultiCheckpointManager(managers={manager_names}, "
            f"save_strategy='{self.save_strategy}', "
            f"load_strategy='{self.load_strategy}', "
            f"enabled={self.enabled})"
        )
