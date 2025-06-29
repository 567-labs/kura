from kura.base_classes import (
    BaseEmbeddingModel,
    BaseClusteringMethod,
    BaseClusterDescriptionModel,
)
from kura.base_classes import BaseCheckpointManager
from kura.embedding import embed_summaries, OpenAIEmbeddingModel
from kura.types.summarisation import ConversationSummary
from kura.types.cluster import Cluster, GeneratedCluster
import logging
import math
from typing import Union, cast, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from instructor.models import KnownModelName
    from instructor import AsyncInstructor

import numpy as np
import asyncio

from asyncio import Semaphore
from rich.console import Console

logger = logging.getLogger(__name__)

# ============================================================================
# Cluster Prompt
# ============================================================================

DEFAULT_CLUSTER_PROMPT = """
You are tasked with summarizing a group of related statements into a short, precise, and accurate description and name. Your goal is to create a concise summary that captures the essence of these statements and distinguishes them from other similar groups of statements.

Summarize all the statements into a clear, precise, two-sentence description in the past tense. Your summary should be specific to this group and distinguish it from the contrastive answers of the other groups.

After creating the summary, generate a short name for the group of statements. This name should be at most ten words long (perhaps less) and be specific but also reflective of most of the statements (rather than reflecting only one or two).

The name should distinguish this group from the contrastive examples. For instance, "Write fantasy sexual roleplay with octopi and monsters", "Generate blog spam for gambling websites", or "Assist with high school math homework" would be better and more actionable than general terms like "Write erotic content" or "Help with homework". Be as descriptive as possible and assume neither good nor bad faith. Do not hesitate to identify and describe socially harmful or sensitive topics specifically; specificity is necessary for monitoring.

Present your output in the following format:
<summary> [Insert your two-sentence summary here] </summary>
<name> [Insert your generated short name here] </name>

The names you propose must follow these requirements:
- The cluster name should be a sentence in the imperative that captures the user's request. For example, 'Brainstorm ideas for a birthday party' or 'Help me find a new job.'
- Create names that are specific enough to be meaningful, but not so specific that they can't meaningfully represent many different statements.
- Avoid overly general or vague terms, and do not hesitate to describe socially harmful or sensitive topics (in fact, clusters that clearly describe harmful behavior are slightly preferred); specificity is necessary for observability and enforcement.
- Ensure that the cluster name is distinct from the contrastive examples.
- Use clear, concise, and descriptive language for the cluster name.

Below are the related statements:
<positive_examples>
{% for item in positive_examples %}{{ item }}
{% endfor %}
</positive_examples>

For context, here are statements from nearby groups that are NOT part of the group you're summarizing:
<contrastive_examples>
{% for item in contrastive_examples %}{{ item }}
{% endfor %}
</contrastive_examples>

Do not elaborate beyond what you say in the tags. Remember to analyze both the statements and the contrastive statements carefully to ensure your summary and name accurately represent the specific group while distinguishing it from others. The cluster name should be a sentence in the imperative that captures the user's request. For example, 'Brainstorm ideas for a birthday party' or 'Help me find a new job.'
"""


# ============================================================================
# Cluster Model
# ============================================================================


class ClusterDescriptionModel(BaseClusterDescriptionModel):
    """
    Model for generating cluster descriptions using LLMs.

    Similar to SummaryModel, this handles the LLM interaction for generating
    cluster names and descriptions with configurable parameters.
    """

    def __init__(
        self,
        model: Union[str, "KnownModelName"] = "openai/gpt-4o-mini",
        max_concurrent_requests: int = 50,
        temperature: float = 0.2,
        checkpoint_filename: str = "clusters",
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
        prompt: str = DEFAULT_CLUSTER_PROMPT,
        max_contrastive_examples: int = 10,
    ) -> List[Cluster]:
        """Generate clusters from a mapping of cluster IDs to summaries."""
        import instructor

        self.sem = Semaphore(self.max_concurrent_requests)
        self.client = instructor.from_provider(self.model, async_client=True)

        if not self.console:
            # Simple processing without rich display
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

        return await self._generate_clusters_with_console(
            cluster_id_to_summaries,
            max_contrastive_examples,
            prompt,
        )

    async def generate_cluster_description(
        self,
        summaries: List[ConversationSummary],
        contrastive_examples: List[ConversationSummary],
        semaphore: Semaphore,
        client: "AsyncInstructor",
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

    async def _generate_clusters_with_console(
        self,
        cluster_id_to_summaries: Dict[int, List[ConversationSummary]],
        max_contrastive_examples: int,
        prompt: str,
    ) -> List[Cluster]:
        """
        Generate clusters with full-screen Rich console display showing progress and latest results.
        """
        from rich.live import Live
        from rich.layout import Layout
        from rich.panel import Panel
        from rich.text import Text
        from rich.progress import (
            Progress,
            SpinnerColumn,
            TextColumn,
            TaskProgressColumn,
            TimeRemainingColumn,
        )

        clusters = []
        completed_clusters = []
        max_preview_items = 3
        total_clusters = len(cluster_id_to_summaries)

        # Create full-screen layout
        layout = Layout()
        layout.split_column(Layout(name="progress", size=3), Layout(name="preview"))

        # Create progress bar
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=self.console,
        )
        task_id = progress.add_task("", total=total_clusters)
        layout["progress"].update(progress)

        def update_preview_display():
            if completed_clusters:
                preview_text = Text()

                for cluster in completed_clusters[-max_preview_items:]:  # Show latest 3
                    preview_text.append(f"Name: {cluster.name}\n", style="bold green")
                    preview_text.append(
                        f"Description: {cluster.description[:100]}{'...' if len(cluster.description) > 100 else ''}\n",
                        style="white",
                    )
                    preview_text.append(
                        f"Conversations: {len(cluster.chat_ids)}\n\n",
                        style="yellow",
                    )

                layout["preview"].update(
                    Panel(
                        preview_text,
                        title=f"[green]Generated Clusters ({len(completed_clusters)}/{total_clusters})",
                        border_style="green",
                    )
                )
            else:
                layout["preview"].update(
                    Panel(
                        Text("Waiting for clusters...", style="dim"),
                        title="[yellow]Generated Clusters (0/0)",
                        border_style="yellow",
                    )
                )

        # Initialize preview display
        update_preview_display()

        with Live(layout, console=self.console, refresh_per_second=4):
            # Prepare tasks for each cluster
            tasks = []
            for cluster_id, summaries in cluster_id_to_summaries.items():
                coro = self.generate_cluster_description(
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
                tasks.append(coro)

            # Use asyncio.as_completed to show results as they finish
            for i, coro in enumerate(asyncio.as_completed(tasks)):
                try:
                    cluster = await coro
                    clusters.append(cluster)
                    completed_clusters.append(cluster)

                    # Update progress
                    progress.update(task_id, completed=i + 1)

                    # Update preview display
                    update_preview_display()

                except Exception as e:
                    logger.error(f"Failed to generate cluster: {e}")
                    # Still update progress on error
                    progress.update(task_id, completed=i + 1)
                    update_preview_display()

        return clusters


# ============================================================================
# Clustering Methods
# ============================================================================


class KmeansClusteringModel(BaseClusteringMethod):
    def __init__(self, clusters_per_group: int = 10):
        self.clusters_per_group = clusters_per_group
        logger.info(
            f"Initialized KmeansClusteringModel with clusters_per_group={clusters_per_group}"
        )

    def cluster(
        self, items: list[dict[str, Union[ConversationSummary, list[float]]]]
    ) -> dict[int, list[ConversationSummary]]:
        """
        We perform a clustering here using an embedding defined on each individual item.

        We assume that the item is passed in as a dictionary with

        - its relevant embedding stored in the "embedding" key.
        - the item itself stored in the "item" key.

        {
            "embedding": list[float],
            "item": any,
        }
        """
        from sklearn.cluster import KMeans

        if not items:
            logger.warning("Empty items list provided to cluster method")
            raise ValueError("Empty items list provided to cluster method")

        logger.info(f"Starting K-means clustering of {len(items)} items")

        try:
            embeddings = [item["embedding"] for item in items]  # pyright: ignore
            data = [item["item"] for item in items]
            n_clusters = math.ceil(len(data) / self.clusters_per_group)

            logger.debug(
                f"Calculated {n_clusters} clusters for {len(data)} items (target: {self.clusters_per_group} items per cluster)"
            )

            X = np.array(embeddings)
            logger.debug(f"Created embedding matrix of shape {X.shape}")

            kmeans = KMeans(n_clusters=n_clusters)
            cluster_labels = kmeans.fit_predict(X)

            logger.debug(
                f"K-means clustering completed, assigned {len(set(cluster_labels))} unique cluster labels"
            )

            result = {
                i: [data[j] for j in range(len(data)) if cluster_labels[j] == i]
                for i in range(n_clusters)
            }

            # Log cluster size distribution
            cluster_sizes = [len(cluster_items) for cluster_items in result.values()]
            logger.info(
                f"K-means clustering completed: {len(result)} clusters created with sizes {cluster_sizes}"
            )
            logger.debug(
                f"Cluster size stats - min: {min(cluster_sizes)}, max: {max(cluster_sizes)}, avg: {sum(cluster_sizes) / len(cluster_sizes):.1f}"
            )

            return cast(dict[int, list[ConversationSummary]], result)

        except Exception as e:
            logger.error(
                f"Failed to perform K-means clustering on {len(items)} items: {e}"
            )
            raise


# ============================================================================
# Utility Functions
# ============================================================================


def get_contrastive_examples(
    cluster_id: int,
    cluster_id_to_summaries: Dict[int, List[ConversationSummary]],
    max_contrastive_examples: int = 10,
) -> List[ConversationSummary]:
    """Get contrastive examples from other clusters to help distinguish this cluster.

    Args:
        cluster_id: The id of the cluster to get contrastive examples for
        cluster_id_to_summaries: A dictionary of cluster ids to their summaries
        max_contrastive_examples: The number of contrastive examples to return. Defaults to 10.

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


async def generate_base_clusters_from_conversation_summaries(
    summaries: List[ConversationSummary],
    embedding_model: Optional[BaseEmbeddingModel] = None,
    clustering_method: Optional[BaseClusteringMethod] = None,
    clustering_model: Optional[BaseClusterDescriptionModel] = None,
    checkpoint_manager: Optional[BaseCheckpointManager] = None,
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
        raise ValueError("Empty summaries list provided")

    # Initialize default models if not provided
    if embedding_model is None:
        embedding_model = OpenAIEmbeddingModel()
    if clustering_method is None:
        clustering_method = KmeansClusteringModel()
    if clustering_model is None:
        clustering_model = ClusterDescriptionModel()

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
