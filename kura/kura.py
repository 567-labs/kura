from kura.dimensionality import HDBUMAP
from kura.types import Cluster
from kura.embedding import OpenAIEmbeddingModel
from kura.summarisation import SummaryModel
from kura.meta_cluster import MetaClusterModel
from kura.cluster import ClusterDescriptionModel
from kura.visualization import ClusterVisualizer
from kura.base_classes import (
    BaseEmbeddingModel,
    BaseSummaryModel,
    BaseClusterDescriptionModel,
    BaseMetaClusterModel,
    BaseDimensionalityReduction,
)
from typing import Union, Optional, TypeVar, TYPE_CHECKING

if TYPE_CHECKING:
    from rich.console import Console as RichConsole
else:
    RichConsole = None
import os
from pydantic import BaseModel
from kura.types.dimensionality import ProjectedCluster
from kura.types import ConversationSummary

# Try to import Rich, fall back gracefully if not available
try:
    from rich.console import Console

    RICH_AVAILABLE = True
except ImportError:
    Console = None
    RICH_AVAILABLE = False

T = TypeVar("T", bound=BaseModel)


class Kura:
    """Main class for the Kura conversation analysis pipeline.

    Kura is a tool for analyzing conversation data using a multi-step process of
    summarization, embedding, clustering, meta-clustering, and visualization.
    This class coordinates the entire pipeline and manages checkpointing.

    Note: This class-based approach is deprecated.
    Please use the procedural API functions instead.

    Attributes:
        embedding_model: Model for converting text to vector embeddings
        summarisation_model: Model for generating summaries from conversations
        cluster_model: Model for initial clustering of summaries
        meta_cluster_model: Model for creating hierarchical clusters
        dimensionality_reduction: Model for projecting clusters to 2D space
        checkpoint_dir: Directory for saving intermediate results
    """

    def __init__(
        self,
        embedding_model: Union[BaseEmbeddingModel, None] = None,
        summarisation_model: Union[BaseSummaryModel, None] = None,
        cluster_model: Union[BaseClusterDescriptionModel, None] = None,
        meta_cluster_model: Union[BaseMetaClusterModel, None] = None,
        dimensionality_reduction: BaseDimensionalityReduction = HDBUMAP(),
        checkpoint_dir: str = "./checkpoints",
        conversation_checkpoint_name: str = "conversations.json",
        disable_checkpoints: bool = False,
        console: Optional["RichConsole"] = None,
        disable_progress: bool = False,
        **kwargs,  # For future use
    ):
        """Initialize a new Kura instance with custom or default components.

        Args:
            embedding_model: Model to convert text to vector embeddings (default: OpenAIEmbeddingModel)
            summarisation_model: Model to generate summaries from conversations (default: SummaryModel)
            cluster_model: Model for initial clustering (default: ClusterModel)
            meta_cluster_model: Model for hierarchical clustering (default: MetaClusterModel)
            dimensionality_reduction: Model for 2D projection (default: HDBUMAP)
            checkpoint_dir: Directory for saving intermediate results (default: "./checkpoints")
            conversation_checkpoint_name: Filename for conversations checkpoint (default: "conversations.json")
            disable_checkpoints: Whether to disable checkpoint loading/saving (default: False)
            console: Optional Rich console instance to use for output (default: None, will create if Rich is available)
            disable_progress: Whether to disable all progress bars for cleaner output (default: False)
            **kwargs: Additional keyword arguments passed to model constructors

        Note:
            Checkpoint filenames for individual processing steps (summaries, clusters, meta-clusters,
            dimensionality reduction) are now defined as properties in their respective base classes
            rather than constructor arguments.
        """
        from warnings import warn

        warn(
            "Kura is deprecated. Please use the procedural API functions instead.",
            DeprecationWarning,
        )

        # Initialize Rich console if available and not provided
        if console is None and RICH_AVAILABLE and not disable_progress and Console:
            self.console = Console()
        else:
            self.console = console

        # Store progress settings
        self.disable_progress = disable_progress

        # Initialize models with console
        if embedding_model is None:
            self.embedding_model = OpenAIEmbeddingModel()
        else:
            self.embedding_model = embedding_model

        console_to_pass = self.console if not disable_progress else None

        if summarisation_model is None:
            self.summarisation_model = SummaryModel(console=console_to_pass, **kwargs)
        else:
            self.summarisation_model = summarisation_model

        if cluster_model is None:
            self.cluster_model = ClusterDescriptionModel(
                console=console_to_pass, **kwargs
            )
        else:
            self.cluster_model = cluster_model

        if meta_cluster_model is None:
            # Pass max_clusters to MetaClusterModel if provided
            self.meta_cluster_model = MetaClusterModel(
                console=console_to_pass, **kwargs
            )
        else:
            self.meta_cluster_model = meta_cluster_model
        self.dimensionality_reduction = dimensionality_reduction

        # Define Checkpoints
        self.checkpoint_dir = checkpoint_dir

        # Helper to construct checkpoint paths
        def _checkpoint_path(filename: str) -> str:
            return os.path.join(self.checkpoint_dir, filename)

        self.conversation_checkpoint_name = _checkpoint_path(
            conversation_checkpoint_name
        )
        self.disable_checkpoints = disable_checkpoints

        # Initialize visualizer
        self._visualizer = None

    @property
    def summary_checkpoint_path(self) -> str:
        """Get the checkpoint path for summaries based on the summarisation model."""
        return os.path.join(
            self.checkpoint_dir, self.summarisation_model.checkpoint_filename
        )

    @property
    def cluster_checkpoint_path(self) -> str:
        """Get the checkpoint path for clusters based on the cluster model."""
        return os.path.join(self.checkpoint_dir, self.cluster_model.checkpoint_filename)

    @property
    def meta_cluster_checkpoint_path(self) -> str:
        """Get the checkpoint path for meta-clusters based on the meta-cluster model."""
        return os.path.join(
            self.checkpoint_dir, self.meta_cluster_model.checkpoint_filename
        )

    @property
    def dimensionality_checkpoint_path(self) -> str:
        """Get the checkpoint path for dimensionality reduction based on the dimensionality model."""
        return os.path.join(
            self.checkpoint_dir, self.dimensionality_reduction.checkpoint_filename
        )

    def load_checkpoint(
        self, checkpoint_path: str, response_model: type[T]
    ) -> Union[list[T], None]:
        """Load data from a checkpoint file if it exists.

        Args:
            checkpoint_path: Path to the checkpoint file
            response_model: Pydantic model class for deserializing the data

        Returns:
            List of model instances if checkpoint exists, None otherwise
        """
        if not self.disable_checkpoints:
            if os.path.exists(checkpoint_path):
                print(
                    f"Loading checkpoint from {checkpoint_path} for {response_model.__name__}"
                )
                with open(checkpoint_path, "r") as f:
                    return [response_model.model_validate_json(line) for line in f]
        return None

    def save_checkpoint(self, checkpoint_path: str, data: list[T]) -> None:
        """Save data to a checkpoint file.

        Args:
            checkpoint_path: Path to the checkpoint file
            data: List of model instances to save
        """
        if not self.disable_checkpoints:
            with open(checkpoint_path, "w") as f:
                for item in data:
                    f.write(item.model_dump_json() + "\n")

    def setup_checkpoint_dir(self):
        """Set up the checkpoint directory.

        Creates the checkpoint directory if it doesn't exist.
        If override_checkpoint_dir is True, removes and recreates the directory.
        """
        if self.disable_checkpoints:
            return

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

    async def reduce_clusters(self, clusters: list[Cluster]) -> list[Cluster]:
        """Reduce clusters into a hierarchical structure.

        Iteratively combines similar clusters until the number of root clusters
        is less than or equal to the meta_cluster_model's max_clusters.

        Args:
            clusters: List of initial clusters

        Returns:
            List of clusters with hierarchical structure
        """
        checkpoint_items = self.load_checkpoint(
            self.meta_cluster_checkpoint_path, Cluster
        )
        if checkpoint_items:
            return checkpoint_items

        root_clusters = clusters

        print(f"Starting with {len(root_clusters)} clusters")

        while len(root_clusters) > self.meta_cluster_model.max_clusters:
            # We get the updated list of clusters
            new_current_level = await self.meta_cluster_model.reduce_clusters(
                root_clusters
            )

            # These are the new root clusters that we've generated
            root_clusters = [c for c in new_current_level if c.parent_id is None]

            # We then remove outdated versions of clusters
            old_cluster_ids = {rc.id for rc in new_current_level if rc.parent_id}
            clusters = [c for c in clusters if c.id not in old_cluster_ids]

            # We then add the new clusters to the list
            clusters.extend(new_current_level)

            print(f"Reduced to {len(root_clusters)} clusters")

        self.save_checkpoint(self.meta_cluster_checkpoint_path, clusters)
        return clusters

    async def generate_base_clusters(
        self, summaries: list[ConversationSummary]
    ) -> list[Cluster]:
        """Generate base clusters from summaries.

        Uses the cluster_model to group similar summaries into clusters.
        Loads from checkpoint if available.

        Args:
            summaries: List of conversation summaries

        Returns:
            List of base clusters
        """
        checkpoint_items = self.load_checkpoint(self.cluster_checkpoint_path, Cluster)
        if checkpoint_items:
            return checkpoint_items

        clusters = []
        return clusters

    async def reduce_dimensionality(
        self, clusters: list[Cluster]
    ) -> list[ProjectedCluster]:
        """Reduce dimensions of clusters for visualization.

        Uses dimensionality_reduction to project clusters to 2D space.
        Loads from checkpoint if available.

        Args:
            clusters: List of clusters to project

        Returns:
            List of projected clusters with 2D coordinates
        """
        checkpoint_items = self.load_checkpoint(
            self.dimensionality_checkpoint_path, ProjectedCluster
        )
        if checkpoint_items:
            return checkpoint_items

        dimensionality_reduced_clusters = (
            await self.dimensionality_reduction.reduce_dimensionality(clusters)
        )

        self.save_checkpoint(
            self.dimensionality_checkpoint_path, dimensionality_reduced_clusters
        )
        return dimensionality_reduced_clusters

    @property
    def visualizer(self) -> ClusterVisualizer:
        """Get or create the cluster visualizer."""
        if self._visualizer is None:
            self._visualizer = ClusterVisualizer(self)
        return self._visualizer

    def visualise_clusters(self):
        """Print a hierarchical visualization of clusters to the terminal.

        Delegates to the ClusterVisualizer for the actual visualization.
        """
        self.visualizer.visualise_clusters()

    def visualise_clusters_enhanced(self):
        """Print an enhanced hierarchical visualization of clusters.

        Delegates to the ClusterVisualizer for the actual visualization.
        """
        self.visualizer.visualise_clusters_enhanced()

    def visualise_clusters_rich(self):
        """Print a rich-formatted hierarchical visualization using Rich library.

        Delegates to the ClusterVisualizer for the actual visualization.
        """
        self.visualizer.visualise_clusters_rich()
