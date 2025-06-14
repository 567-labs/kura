from .checkpoint import CheckpointManager
from .checkpoints import MultiCheckpointManager
from .summarisation import SummaryModel, summarise_conversations
from .cluster import (
    ClusterDescriptionModel,
    generate_base_clusters_from_conversation_summaries,
)
from .v1.kura import (
    reduce_clusters_from_base_clusters,
    reduce_dimensionality_from_clusters,
)
from .meta_cluster import MetaClusterModel

from .types import Conversation
from .k_means import KmeansClusteringMethod, MiniBatchKmeansClusteringMethod
from .hdbscan import HDBSCANClusteringMethod
from .v1.visualization import (
    visualise_pipeline_results,
    visualise_clusters_rich,
    visualise_clusters_enhanced,
    visualise_clusters,
)

# Import ParquetCheckpointManager from checkpoints module if available
try:
    from .checkpoints.parquet import ParquetCheckpointManager

    PARQUET_AVAILABLE = True
except ImportError:
    ParquetCheckpointManager = None
    PARQUET_AVAILABLE = False

try:
    from .checkpoints.hf_dataset import HFDatasetCheckpointManager

    HF_AVAILABLE = True
except ImportError:
    HFDatasetCheckpointManager = None
    HF_AVAILABLE = False


__all__ = [
    "SummaryModel",
    "ClusterDescriptionModel",
    "Conversation",
    "MetaClusterModel",
    "CheckpointManager",
    "MultiCheckpointManager",
    "KmeansClusteringMethod",
    "MiniBatchKmeansClusteringMethod",
    "HDBSCANClusteringMethod",
    # Procedural Methods
    "summarise_conversations",
    "generate_base_clusters_from_conversation_summaries",
    "reduce_clusters_from_base_clusters",
    "reduce_dimensionality_from_clusters",
    # Visualisation
    "visualise_pipeline_results",
    "visualise_clusters_rich",
    "visualise_clusters_enhanced",
    "visualise_clusters",
]

# Add ParquetCheckpointManager to __all__ if available
if PARQUET_AVAILABLE:
    __all__.append("ParquetCheckpointManager")

if HF_AVAILABLE:
    __all__.append("HFDatasetCheckpointManager")

__version__ = "1.0.0"
