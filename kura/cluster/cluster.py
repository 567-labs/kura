from kura.base_classes import BaseEmbeddingModel, BaseClusteringMethod
from kura.checkpoint import CheckpointManager
from kura.embedding.embedding import embed_summaries
from kura.embedding.models import OpenAIEmbeddingModel
from kura.cluster.models import KmeansClusteringModel
from kura.types.summarisation import ConversationSummary
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


async def cluster_conversations(
    summaries: List[ConversationSummary],
    embedding_model: BaseEmbeddingModel = OpenAIEmbeddingModel(),
    clustering_model: BaseClusteringMethod = KmeansClusteringModel(),
    checkpoint_manager: CheckpointManager = None,
    **kwargs,
) -> Dict[int, List[ConversationSummary]]:
    """
    Cluster conversation summaries using embeddings.

    Args:
        summaries: List of conversation summaries to cluster
        embedding_model: Model for generating embeddings (defaults to OpenAI)
        clustering_model: Clustering algorithm (defaults to K-means)
        **kwargs: Additional parameters for clustering model

    Returns:
        Dictionary mapping cluster IDs to lists of conversation summaries
    """
    if not summaries:
        logger.warning("Empty summaries list provided")
        return {}

    logger.info(f"Clustering {len(summaries)} conversation summaries")

    # Embed the summaries
    embedded_items = await embed_summaries(summaries, embedding_model)

    # Perform clustering
    clusters = clustering_model.cluster(embedded_items)

    logger.info(f"Created {len(clusters)} clusters")

    return clusters
