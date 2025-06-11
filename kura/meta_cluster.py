from __future__ import annotations

from kura.base_classes import (
    BaseMetaClusterModel,
    BaseEmbeddingModel,
    BaseClusteringMethod,
)
import math
from kura.types.cluster import Cluster, GeneratedCluster
from kura.embedding import OpenAIEmbeddingModel
import instructor
from asyncio import Semaphore
from pydantic import BaseModel, field_validator, ValidationInfo
import re
from thefuzz import fuzz
import asyncio
import logging
from typing import Optional, Union

# Rich imports handled by Kura base class
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rich.console import Console

logger = logging.getLogger(__name__)


class CandidateClusters(BaseModel):
    candidate_cluster_names: list[str]

    @field_validator("candidate_cluster_names")
    def validate_candidate_cluster_names(cls, v: list[str]) -> list[str]:
        if len(v) == 0:
            raise ValueError("Candidate cluster names must be a non-empty list")

        v = [label.strip() for label in v]
        v = [label[:-1] if label.endswith(".") else label for label in v]

        return [re.sub(r"\\{1,}", "", label.replace('"', "")) for label in v]


class ClusterLabel(BaseModel):
    higher_level_cluster: str

    @field_validator("higher_level_cluster")
    def validate_higher_level_cluster(cls, v: str, info: ValidationInfo) -> str:
        candidate_clusters = info.context["candidate_clusters"]  # pyright: ignore

        # Exact match check
        if v in candidate_clusters:
            return v

        # Fuzzy match check with 90% similarity threshold
        for candidate in candidate_clusters:
            similarity = fuzz.ratio(v, candidate)
            if similarity >= 90:  # 90% similarity threshold
                return candidate

        # If no match found
        raise ValueError(
            f"""
            Invalid higher-level cluster: |{v}|

            Valid clusters are:
            {", ".join(f"|{c}|" for c in candidate_clusters)}
            """
        )
        return v


class MetaClusterModel(BaseMetaClusterModel):
    @property
    def checkpoint_filename(self) -> str:
        """The filename to use for checkpointing this model's output."""
        return "meta_clusters"

    def __init__(
        self,
        max_concurrent_requests: int = 50,
        model: str = "openai/gpt-4o-mini",
        embedding_model: BaseEmbeddingModel = OpenAIEmbeddingModel(),
        clustering_model: Union[BaseClusteringMethod, None] = None,
        max_clusters: int = 10,
        console: Optional["Console"] = None,
        **kwargs,  # For future use
    ):
        if clustering_model is None:
            from kura.cluster import KmeansClusteringModel

            clustering_model = KmeansClusteringModel(12)

        self.max_concurrent_requests = max_concurrent_requests
        self.sem = Semaphore(max_concurrent_requests)
        self.client = instructor.from_provider(model, async_client=True)
        self.console = console
        self.max_clusters = max_clusters

        if embedding_model is None:
            embedding_model = OpenAIEmbeddingModel()

        self.embedding_model = embedding_model
        self.clustering_model = clustering_model
        self.model = model
        self.console = console

        logger.info(
            f"Initialized MetaClusterModel with model={model}, max_concurrent_requests={max_concurrent_requests}, embedding_model={type(embedding_model).__name__}, clustering_model={type(clustering_model).__name__}, max_clusters={max_clusters}"
        )

        # Debug: Check if console is set
        if self.console:
            logger.debug(f"Console is set to {type(self.console)}")
        else:
            logger.debug("Console is None - Rich progress bars will not be available")

    async def _gather_with_progress(
        self,
        tasks,
        desc: str = "Processing",
        disable: bool = False,
        show_preview: bool = False,
    ):
        """Helper method to run async gather with Rich progress bar if available, otherwise tqdm."""
        if self.console and not disable:
            try:
                from rich.progress import (
                    Progress,
                    SpinnerColumn,
                    TextColumn,
                    BarColumn,
                    TaskProgressColumn,
                    TimeRemainingColumn,
                )
                from rich.live import Live
                from rich.layout import Layout
                from rich.panel import Panel
                from rich.text import Text
                from rich.errors import LiveError

                # Check if a Live display is already active by trying to get the current live instance
                try:
                    # Try to access the console's current live instance
                    if (
                        hasattr(self.console, "_live")
                        and self.console._live is not None
                    ):
                        show_preview = (
                            False  # Disable preview if Live is already active
                        )
                except AttributeError:
                    pass  # Console doesn't have _live attribute, that's fine

                if show_preview:
                    # Use Live display with progress and preview buffer
                    layout = Layout()
                    layout.split_column(
                        Layout(name="progress", size=3), Layout(name="preview")
                    )

                    preview_buffer = []
                    max_preview_items = 3

                    # Create progress with cleaner display
                    progress = Progress(
                        SpinnerColumn(),
                        TextColumn("[progress.description]{task.description}"),
                        BarColumn(),
                        TaskProgressColumn(),
                        TimeRemainingColumn(),
                        console=self.console,
                    )
                    task_id = progress.add_task(f"[cyan]{desc}...", total=len(tasks))
                    layout["progress"].update(progress)

                    try:
                        with Live(layout, console=self.console, refresh_per_second=4):
                            completed_tasks = []
                            for i, task in enumerate(asyncio.as_completed(tasks)):
                                result = await task
                                completed_tasks.append(result)
                                progress.update(task_id, completed=i + 1)

                                # Handle different result types
                                if isinstance(result, list):
                                    # For operations that return lists of clusters
                                    for item in result:
                                        if (
                                            hasattr(item, "name")
                                            and hasattr(item, "description")
                                            and item.parent_id is None
                                        ):
                                            preview_buffer.append(item)
                                            if len(preview_buffer) > max_preview_items:
                                                preview_buffer.pop(0)
                                elif hasattr(result, "name") and hasattr(
                                    result, "description"
                                ):
                                    # For operations that return single clusters
                                    preview_buffer.append(result)
                                    if len(preview_buffer) > max_preview_items:
                                        preview_buffer.pop(0)

                                # Update preview display if we have clusters
                                if preview_buffer:
                                    preview_text = Text()
                                    for j, cluster in enumerate(preview_buffer):
                                        preview_text.append(
                                            "Meta Cluster: ", style="bold magenta"
                                        )
                                        preview_text.append(
                                            f"{cluster.name[:80]}...\n",
                                            style="bold white",
                                        )
                                        preview_text.append(
                                            "Description: ", style="bold cyan"
                                        )
                                        preview_text.append(
                                            f"{cluster.description[:100]}...\n\n",
                                            style="dim white",
                                        )

                                    layout["preview"].update(
                                        Panel(
                                            preview_text,
                                            title=f"[magenta]Recent Meta Clusters ({len(preview_buffer)}/{max_preview_items})",
                                            border_style="magenta",
                                        )
                                    )

                            return completed_tasks
                    except LiveError:
                        # If Rich Live fails (e.g., another Live is active), fall back to simple progress
                        with progress:
                            completed_tasks = []
                            for i, task in enumerate(asyncio.as_completed(tasks)):
                                result = await task
                                completed_tasks.append(result)
                                progress.update(task_id, completed=i + 1)
                            return completed_tasks
                else:
                    # Regular progress bar without preview (or when Live is already active)
                    progress = Progress(
                        SpinnerColumn(),
                        TextColumn("[progress.description]{task.description}"),
                        BarColumn(),
                        TaskProgressColumn(),
                        TimeRemainingColumn(),
                        console=self.console,
                    )

                    with progress:
                        task_id = progress.add_task(
                            f"[cyan]{desc}...", total=len(tasks)
                        )

                        completed_tasks = []
                        for i, task in enumerate(asyncio.as_completed(tasks)):
                            result = await task
                            completed_tasks.append(result)
                            progress.update(task_id, completed=i + 1)

                        return completed_tasks

            except (ImportError, LiveError):  # type: ignore
                # Rich not available or Live error, run silently
                return await asyncio.gather(*tasks)
        else:
            # No console, run silently
            return await asyncio.gather(*tasks)

    async def generate_candidate_clusters(
        self, clusters: list[Cluster], sem: Semaphore
    ) -> list[str]:
        async with sem:
            resp = await self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": """
                You are tasked with creating higher-level cluster names based on a given list of clusters and their descriptions. Your goal is to come up with broader categories that could encompass one or more of the provided clusters

                First, review the list of clusters and their descriptions:
                <cluster_list>
                    {% for cluster in clusters %}
                    <cluster>{{ cluster.name }}: {{ cluster.description }}</cluster>
                    {% endfor %}
                </cluster_list>

                Your task is to create at most {{ desired_number }} higher-level cluster names that could potentially include one or more of the provided clusters. These higher-level clusters should represent broader categories or themes that emerge from the given clusters, while remaining as specific as possible. If there are many clusters with a specific theme, ensure that the higher-level cluster name remains the maximum level of specificity. You are helping to organize user behavior data in order to improve safety, monitoring, and observability. You can generate less than {{ desired_number }} names if you feel that fewer are appropriate and accurately capture the clusters.

                Guidelines for creating higher-level clusters names
                1. Analyze the themes, topics or characteristics common to multiple clusters.
                2. Create names that are specific enough to be meaningful but but not so specific that they can't meaningfully represent many different clusters. Avoid overly general or vague terms, and do not hesitate to describe socially harmful or sensitive topics (in fact, clusters that clearly describe harmful behavior are slightly preferred); specificity is necessary for observability and enforcement.
                3. Ensure that the higher-level cluster names are distinct from one another.
                4. Use clear, concise, and descriptive language for the cluster names. Assume neither good nor bad faith for the content in the clusters.

                Think about the relationships between the given clusters and potential overarching themes.

                Focus on creating meaningful, distinct and precise ( but not overly specific ) higher-level cluster names that could encompass multiple sub-clusters.
                """.strip(),
                    },
                ],
                response_model=CandidateClusters,
                context={
                    "clusters": clusters,
                    "desired_number": math.ceil(len(clusters) / 2)
                    if len(clusters)
                    >= 3  # If we have two clusters we just merge them tbh
                    else 1,
                },
                max_retries=3,
            )
            return resp.candidate_cluster_names

    async def label_cluster(self, cluster: Cluster, candidate_clusters: list[str]):
        async with self.sem:
            resp = await self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": """
You are tasked with categorizing a specific cluster into one of the provided higher-level clusters for observability, monitoring, and content moderation. Your goal is to determine which higher-level cluster best fits the given specific cluster based on its name and description.

First, here are the ONLY valid higher-level clusters you may select from:
<higher_level_clusters>
{% for cluster in candidate_clusters %}
<higher_level_cluster>{{ cluster }}</higher_level_cluster>
{% endfor %}
</higher_level_clusters>

Here is the specific cluster to categorize:
<specific_cluster>
Name: {{ cluster.name }}
Description: {{ cluster.description }}
</specific_cluster>

RULES:
1. You MUST select EXACTLY ONE higher-level cluster from the provided list
2. You MUST output the higher-level cluster name EXACTLY as written - no modifications allowed
3. You MUST NOT create new cluster names or combinations
4. You MUST NOT output any additional text or explanations
5. You MUST NOT use partial matches or approximate names

CLASSIFICATION PROCESS:
1. First, record the exact list of valid higher-level clusters
2. Read the specific cluster's name and description carefully
3. Compare the specific cluster's key characteristics against each valid higher-level cluster
4. Select the single most appropriate higher-level cluster that encompasses the specific cluster
5. Verify your selected cluster exactly matches one from the valid list
6. Output ONLY the selected higher-level cluster name, exactly as it appeared in the valid list

Here is the specific cluster to categorize:

<specific_cluster>
Name: {{ cluster.name }}
Description: {{ cluster.description }}
</specific_cluster>

Based on this information, determine the most appropriate higher-level cluster and provide your answer as instructed.
                        """,
                    }
                ],
                response_model=ClusterLabel,
                context={
                    "cluster": cluster,
                    "candidate_clusters": candidate_clusters,
                },
                max_retries=3,
            )
            return {
                "cluster": cluster,
                "label": resp.higher_level_cluster,
            }

    async def rename_cluster_group(self, clusters: list[Cluster]) -> list[Cluster]:
        async with self.sem:
            resp = await self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": """
                        You are tasked with summarizing a group of related cluster names into a short, precise, and accurate overall description and name. Your goal is to create a concise summary that captures the essence of these clusters

                        The cluster name should be at most ten words long (perhaps less) and be specific but also reflective of most of the clusters that comprise them. The cluster name should be a sentence in the imperative that captures the user's request. For example, 'Brainstorm ideas for a birthday party' or 'Help me find a new job are good examples.

                        Do not shy away from describing socially harmful or sensitive topics. For instance, "Write fantasy sexual roleplay with octopi and monsters", "Generate blog spam for gambling websites", or "Assist with high school math homework" would be better than "Write erotic content" or "Help with homework". Be as descriptive as possible and assume neither good nor bad faith. Do not hesitate to identify and describe socially harmful or sensitive topics specifically; specificity is necessary for monitoring.

                        Ensure your summary and name accurately represent the clusters and are specific to the clusters.
                        """,
                    },
                    {
                        "role": "user",
                        "content": """
                        Here are the related cluster names
                        <clusters>
                            {% for cluster in clusters %}
                                <cluster>{{ cluster.name }}: {{ cluster.description }}</cluster>
                            {% endfor %}
                        </clusters>
                        """,
                    },
                ],
                context={"clusters": clusters},
                response_model=GeneratedCluster,
            )

            res = []

            new_cluster = Cluster(
                name=resp.name,
                description=resp.summary,
                slug=resp.slug,
                chat_ids=[
                    chat_id for cluster in clusters for chat_id in cluster.chat_ids
                ],
                parent_id=None,
            )

            res.append(new_cluster)

            for cluster in clusters:
                res.append(
                    Cluster(
                        id=cluster.id,
                        name=cluster.name,
                        description=cluster.description,
                        slug=cluster.slug,
                        chat_ids=cluster.chat_ids,
                        parent_id=new_cluster.id,
                    )
                )

            return res

    async def generate_meta_clusters(
        self, clusters: list[Cluster], show_preview: bool = True
    ) -> list[Cluster]:
        # Use a single Live display for the entire meta clustering operation
        if self.console and show_preview:
            try:
                from rich.progress import (
                    Progress,
                    SpinnerColumn,
                    TextColumn,
                    BarColumn,
                    TaskProgressColumn,
                    TimeRemainingColumn,
                )
                from rich.live import Live
                from rich.layout import Layout
                from rich.panel import Panel
                from rich.text import Text
                from rich.errors import LiveError

                # Create layout for the entire meta clustering operation
                layout = Layout()
                layout.split_column(
                    Layout(
                        name="progress", size=6
                    ),  # More space for multiple progress bars
                    Layout(name="preview"),
                )

                # Create progress display
                progress = Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TaskProgressColumn(),
                    TimeRemainingColumn(),
                    console=self.console,
                )
                layout["progress"].update(progress)

                preview_buffer = []
                max_preview_items = 3

                try:
                    with Live(layout, console=self.console, refresh_per_second=4):
                        # Step 1: Generate candidate clusters
                        candidate_labels = await self.generate_candidate_clusters(
                            clusters, Semaphore(self.max_concurrent_requests)
                        )

                        # Step 2: Label clusters with progress
                        label_task_id = progress.add_task(
                            "[cyan]Labeling clusters...", total=len(clusters)
                        )
                        cluster_labels = []
                        for i, cluster in enumerate(clusters):
                            result = await self.label_cluster(cluster, candidate_labels)
                            cluster_labels.append(result)
                            progress.update(label_task_id, completed=i + 1)

                        # Group clusters by label
                        label_to_clusters = {}
                        for label in cluster_labels:
                            if label["label"] not in label_to_clusters:
                                label_to_clusters[label["label"]] = []
                            label_to_clusters[label["label"]].append(label["cluster"])

                        # Step 3: Rename cluster groups with progress and preview
                        rename_task_id = progress.add_task(
                            "[cyan]Renaming cluster groups...",
                            total=len(label_to_clusters),
                        )
                        new_clusters = []
                        for i, cluster_group in enumerate(label_to_clusters.values()):
                            result = await self.rename_cluster_group(cluster_group)
                            new_clusters.append(result)
                            progress.update(rename_task_id, completed=i + 1)

                            # Update preview with new meta clusters
                            for cluster in result:
                                if (
                                    hasattr(cluster, "name")
                                    and hasattr(cluster, "description")
                                    and cluster.parent_id is None
                                ):
                                    preview_buffer.append(cluster)
                                    if len(preview_buffer) > max_preview_items:
                                        preview_buffer.pop(0)

                            # Update preview display
                            if preview_buffer:
                                preview_text = Text()
                                for j, cluster in enumerate(preview_buffer):
                                    preview_text.append(
                                        "Meta Cluster: ", style="bold magenta"
                                    )
                                    preview_text.append(
                                        f"{cluster.name[:80]}...\n", style="bold white"
                                    )
                                    preview_text.append(
                                        "Description: ", style="bold cyan"
                                    )
                                    preview_text.append(
                                        f"{cluster.description[:100]}...\n\n",
                                        style="dim white",
                                    )

                                layout["preview"].update(
                                    Panel(
                                        preview_text,
                                        title=f"[magenta]Recent Meta Clusters ({len(preview_buffer)}/{max_preview_items})",
                                        border_style="magenta",
                                    )
                                )

                        # Flatten results
                        res = []
                        for new_cluster in new_clusters:
                            res.extend(new_cluster)

                        return res

                except LiveError:
                    # Fall back to the original method without Live display
                    return await self._generate_meta_clusters_fallback(clusters)

            except ImportError:
                # Rich not available, fall back
                return await self._generate_meta_clusters_fallback(clusters)
        else:
            # No console or preview disabled, use original method
            return await self._generate_meta_clusters_fallback(clusters)

    async def _generate_meta_clusters_fallback(
        self, clusters: list[Cluster]
    ) -> list[Cluster]:
        """Fallback method for generate_meta_clusters when Live display is not available"""
        candidate_labels = await self.generate_candidate_clusters(
            clusters, Semaphore(self.max_concurrent_requests)
        )

        cluster_labels = await self._gather_with_progress(
            [self.label_cluster(cluster, candidate_labels) for cluster in clusters],
            desc="Labeling clusters",
            disable=False,
            show_preview=False,  # Disable preview to avoid nested Live displays
        )

        label_to_clusters = {}
        for label in cluster_labels:
            if label["label"] not in label_to_clusters:
                label_to_clusters[label["label"]] = []

            label_to_clusters[label["label"]].append(label["cluster"])

        new_clusters = await self._gather_with_progress(
            [
                self.rename_cluster_group(cluster)
                for cluster in label_to_clusters.values()
            ],
            desc="Renaming cluster groups",
            show_preview=False,  # Disable preview to avoid nested Live displays
        )

        res = []
        for new_cluster in new_clusters:
            res.extend(new_cluster)

        return res

    async def reduce_clusters(self, clusters: list[Cluster]) -> list[Cluster]:
        """
        This takes in a list of existing clusters and generates a few higher order clusters that are more general. This represents a single iteration of the meta clustering process.

        In the event that we have a single cluster, we will just return a new higher level cluster which has the same name as the original cluster. ( This is an edge case which we should definitely handle better )
        """
        if not clusters:
            return []

        if len(clusters) == 1:
            logger.info("Only one cluster, returning it as a meta cluster")
            new_cluster = Cluster(
                name=clusters[0].name,
                description=clusters[0].description,
                slug=clusters[0].slug,
                chat_ids=clusters[0].chat_ids,
                parent_id=None,
            )
            return [new_cluster, clusters[0]]

        texts_to_embed = [str(cluster) for cluster in clusters]

        logger.info(
            f"Embedding {len(texts_to_embed)} clusters for meta-clustering using {type(self.embedding_model).__name__}..."
        )

        cluster_embeddings = await self.embedding_model.embed(texts_to_embed)

        if not cluster_embeddings or len(cluster_embeddings) != len(clusters):
            logger.error(
                "Error: Number of embeddings does not match number of clusters or embeddings are empty for meta-clustering."
            )
            return []

        clusters_and_embeddings = [
            {
                "item": cluster,
                "embedding": embedding,
            }
            for cluster, embedding in zip(clusters, cluster_embeddings)
        ]

        cluster_id_to_clusters: dict[int, list[Cluster]] = (
            self.clustering_model.cluster(clusters_and_embeddings)
        )  # type: ignore

        new_clusters = await self._gather_with_progress(
            [
                self.generate_meta_clusters(
                    cluster_id_to_clusters[cluster_id], show_preview=True
                )
                for cluster_id in cluster_id_to_clusters
            ],
            desc="Generating Meta Clusters",
            show_preview=True,
        )

        res = []
        for new_cluster in new_clusters:
            res.extend(new_cluster)

        return res
