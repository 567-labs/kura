from kura.dimensionality import HDBUMAP
from kura.types import Conversation, Cluster
from kura.embedding import OpenAIEmbeddingModel
from kura.summarisation import SummaryModel
from kura.meta_cluster import MetaClusterModel
from kura.cluster import ClusterModel
from kura.visualization import ClusterVisualizer
from kura.base_classes import (
    BaseEmbeddingModel,
    BaseSummaryModel,
    BaseClusterModel,
    BaseMetaClusterModel,
    BaseDimensionalityReduction,
)
from typing import Union, Optional, TypeVar
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

    For cleaner output without progress bars:
        kura = Kura(disable_progress=True)

    Or to disable Rich console entirely:
        kura = Kura(console=None)

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
        cluster_model: Union[BaseClusterModel, None] = None,
        meta_cluster_model: Union[BaseMetaClusterModel, None] = None,
        dimensionality_reduction: BaseDimensionalityReduction = HDBUMAP(),
        checkpoint_dir: str = "./checkpoints",
        conversation_checkpoint_name: str = "conversations.json",
        disable_checkpoints: bool = False,
        console: Optional["Console"] = None,  # type: ignore
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
            self.cluster_model = ClusterModel(console=console_to_pass, **kwargs)
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
    def visualizer(self) -> ClusterVisualizer:
        """Get or create the cluster visualizer."""
        if self._visualizer is None:
            self._visualizer = ClusterVisualizer(console=self.console)
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
