from kura.base_classes import BaseEmbeddingModel, BaseClusteringMethod, BaseClusterModel
from kura.checkpoint import CheckpointManager
from kura.embedding.embedding import embed_summaries
from kura.embedding.models import OpenAIEmbeddingModel
from kura.cluster.models import KmeansClusteringModel
from kura.cluster.constants import DEFAULT_CLUSTER_PROMPT
from kura.types.summarisation import ConversationSummary
from kura.types.cluster import Cluster, GeneratedCluster
import logging
import numpy as np
import asyncio
import instructor
from instructor.models import KnownModelName
from asyncio import Semaphore
from typing import Dict, List, Optional, Union
from rich.console import Console

logger = logging.getLogger(__name__)


class ClusterModel:
    """
    Model for generating cluster descriptions using LLMs.

    Similar to SummaryModel, this handles the LLM interaction for generating
    cluster names and descriptions with configurable parameters.
    """

    def __init__(
        self,
        model: Union[str, KnownModelName] = "openai/gpt-4o-mini",
        max_concurrent_requests: int = 50,
        temperature: float = 0.2,
        checkpoint_filename: str = "clusters.jsonl",
        console: Optional[Console] = None,
    ):
        """
        Initialize ClusterModel with core configuration.

        Args:
            model: model identifier (e.g., "openai/gpt-4o-mini")
            max_concurrent_requests: Maximum concurrent API requests
            temperature: LLM temperature for generation
            checkpoint_filename: Filename for checkpointing
            console: Rich console for progress tracking
        """
        self.model = model
        self.max_concurrent_requests = max_concurrent_requests
        self.temperature = temperature
        self._checkpoint_filename = checkpoint_filename
        self.console = console

        logger.info(
            f"Initialized ClusterModel with model={model}, max_concurrent_requests={max_concurrent_requests}, temperature={temperature}"
        )

    @property
    def checkpoint_filename(self) -> str:
        """Return the filename to use for checkpointing this model's output."""
        return self._checkpoint_filename

    async def generate_clusters(
        self,
        cluster_id_to_summaries: Dict[int, List[ConversationSummary]],
        max_contrastive_examples: int = 10,
        prompt: str = DEFAULT_CLUSTER_PROMPT,
    ) -> List[Cluster]:
        """Generate clusters from a mapping of cluster IDs to summaries."""
        self.sem = Semaphore(self.max_concurrent_requests)
        self.client = instructor.from_provider(self.model, async_client=True)

        return await asyncio.gather(
            *[
                self.generate_cluster_description(
                    summaries,
                    get_contrastive_examples(
                        cluster_id,
                        cluster_id_to_summaries=cluster_id_to_summaries,
                        max_contrastive_examples=max_contrastive_examples,
                    ),
                    self.sem,
                    self.client,
                    prompt,
                )
                for cluster_id, summaries in cluster_id_to_summaries.items()
            ]
        )

    async def generate_cluster_description(
        self,
        summaries: List[ConversationSummary],
        contrastive_examples: List[ConversationSummary],
        semaphore: Semaphore,
        client: instructor.Instructor,
        prompt: str = DEFAULT_CLUSTER_PROMPT,
    ) -> Cluster:
        """
        Generate a cluster description from summaries with contrastive examples.

        Args:
            summaries: Summaries in this cluster
            contrastive_examples: Examples from other clusters for contrast

        Returns:
            Cluster with generated name and description
        """
        logger.debug(
            f"Generating cluster from {len(summaries)} summaries with {len(contrastive_examples)} contrastive examples"
        )
        async with semaphore:
            try:
                resp = await client.chat.completions.create(
                    messages=[
                        {
                            "role": "system",
                            "content": prompt,
                        },
                    ],
                    response_model=GeneratedCluster,
                    temperature=self.temperature,
                    context={
                        "positive_examples": summaries,
                        "contrastive_examples": contrastive_examples,
                    },
                )

                cluster = Cluster(
                    name=resp.name,
                    description=resp.summary,
                    slug=resp.slug,
                    chat_ids=[item.chat_id for item in summaries],
                    parent_id=None,
                )

                logger.debug(
                    f"Successfully generated cluster '{resp.name}' with {len(summaries)} conversations"
                )
                return cluster

            except Exception as e:
                logger.error(
                    f"Failed to generate cluster from {len(summaries)} summaries: {e}"
                )
                raise


def get_contrastive_examples(
    cluster_id: int,
    cluster_id_to_summaries: Dict[int, List[ConversationSummary]],
    max_contrastive_examples: int = 10,
) -> List[ConversationSummary]:
    """Get contrastive examples from other clusters to help distinguish this cluster.

    Args:
        cluster_id: The id of the cluster to get contrastive examples for
        cluster_id_to_summaries: A dictionary of cluster ids to their summaries
        limit: The number of contrastive examples to return. Defaults to 10.

    Returns:
        List of contrastive examples from other clusters
    """
    other_clusters = [c for c in cluster_id_to_summaries.keys() if c != cluster_id]
    all_examples = []
    for cluster in other_clusters:
        all_examples.extend(cluster_id_to_summaries[cluster])

    logger.debug(
        f"Selecting contrastive examples for cluster {cluster_id}: found {len(all_examples)} examples from {len(other_clusters)} other clusters"
    )

    # If we don't have enough examples, return all of them
    if len(all_examples) <= max_contrastive_examples:
        logger.debug(
            f"Using all {len(all_examples)} available contrastive examples (limit was {max_contrastive_examples})"
        )
        return all_examples

    # Otherwise sample without replacement
    selected = list(
        np.random.choice(all_examples, size=max_contrastive_examples, replace=False)
    )
    logger.debug(
        f"Randomly selected {len(selected)} contrastive examples from {len(all_examples)} available"
    )
    return selected


async def cluster_conversations(
    summaries: List[ConversationSummary],
    embedding_model: BaseEmbeddingModel = OpenAIEmbeddingModel(),
    clustering_method: BaseClusteringMethod = KmeansClusteringModel(),
    clustering_model: BaseClusterModel = ClusterModel(),
    checkpoint_manager: CheckpointManager = None,
    max_contrastive_examples: int = 10,
    prompt: str = DEFAULT_CLUSTER_PROMPT,
    **kwargs,
) -> List[Cluster]:
    """
    Cluster conversation summaries using embeddings.

    Args:
        summaries: List of conversation summaries to cluster
        embedding_model: Model for generating embeddings (defaults to OpenAI)
        clustering_method: Clustering algorithm (defaults to K-means)
        clustering_model: Model for generating cluster descriptions
        checkpoint_manager: Optional checkpoint manager for caching
        max_contrastive_examples: Number of contrastive examples to use
        prompt: Custom prompt for cluster generation
        **kwargs: Additional parameters for clustering model

    Returns:
        List of clusters with generated names and descriptions
    """
    if not summaries:
        logger.warning("Empty summaries list provided")
        return {}

    if checkpoint_manager:
        cached = checkpoint_manager.load_checkpoint(
            clustering_model.checkpoint_filename, Cluster
        )
        if cached:
            logger.info(f"Loaded {len(cached)} clusters from checkpoint")
            return cached

    logger.info(f"Clustering {len(summaries)} conversation summaries")

    # Embed the summaries
    embedded_items = await embed_summaries(summaries, embedding_model)

    # Generate Initial Mapping of Cluster IDs to Summaries
    clusters_id_to_summaries = clustering_method.cluster(embedded_items)

    # Generate Clusters
    clusters = await clustering_model.generate_clusters(
        cluster_id_to_summaries=clusters_id_to_summaries,
        max_contrastive_examples=max_contrastive_examples,
        prompt=prompt,
    )

    if checkpoint_manager:
        checkpoint_manager.save_checkpoint(
            clustering_model.checkpoint_filename, clusters
        )

    return clusters
