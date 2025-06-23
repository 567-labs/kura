# Light imports - these are fast to load
from .types import Conversation

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


def __getattr__(name: str):
    """Lazy loading for heavy imports to improve startup time."""
    # Checkpoint managers
    if name == "CheckpointManager":
        from .checkpoint import CheckpointManager
        return CheckpointManager
    elif name == "MultiCheckpointManager":
        from .checkpoints import MultiCheckpointManager
        return MultiCheckpointManager
    
    # Summary models and functions
    elif name == "SummaryModel":
        from .summarisation import SummaryModel
        return SummaryModel
    elif name == "summarise_conversations":
        from .summarisation import summarise_conversations
        return summarise_conversations
    
    # Clustering models and functions
    elif name == "ClusterDescriptionModel":
        from .cluster import ClusterDescriptionModel
        return ClusterDescriptionModel
    elif name == "generate_base_clusters_from_conversation_summaries":
        from .cluster import generate_base_clusters_from_conversation_summaries
        return generate_base_clusters_from_conversation_summaries
    
    # Clustering methods
    elif name == "KmeansClusteringMethod":
        from .k_means import KmeansClusteringMethod
        return KmeansClusteringMethod
    elif name == "MiniBatchKmeansClusteringMethod":
        from .k_means import MiniBatchKmeansClusteringMethod
        return MiniBatchKmeansClusteringMethod
    elif name == "HDBSCANClusteringMethod":
        from .hdbscan import HDBSCANClusteringMethod
        return HDBSCANClusteringMethod
    
    # Meta clustering
    elif name == "MetaClusterModel":
        from .meta_cluster import MetaClusterModel
        return MetaClusterModel
    
    # V1 functions
    elif name == "reduce_clusters_from_base_clusters":
        from .v1.kura import reduce_clusters_from_base_clusters
        return reduce_clusters_from_base_clusters
    elif name == "reduce_dimensionality_from_clusters":
        from .v1.kura import reduce_dimensionality_from_clusters
        return reduce_dimensionality_from_clusters
    
    # Visualization functions
    elif name == "visualise_pipeline_results":
        from .v1.visualization import visualise_pipeline_results
        return visualise_pipeline_results
    elif name == "visualise_clusters_rich":
        from .v1.visualization import visualise_clusters_rich
        return visualise_clusters_rich
    elif name == "visualise_clusters_enhanced":
        from .v1.visualization import visualise_clusters_enhanced
        return visualise_clusters_enhanced
    elif name == "visualise_clusters":
        from .v1.visualization import visualise_clusters
        return visualise_clusters
    
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


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
