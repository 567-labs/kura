from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Import types for static analysis
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
    
    # Optional imports for type checking
    try:
        from .checkpoints.parquet import ParquetCheckpointManager
    except ImportError:
        ParquetCheckpointManager = None
    
    try:
        from .checkpoints.hf_dataset import HFDatasetCheckpointManager
    except ImportError:
        HFDatasetCheckpointManager = None

# Lazy loading configuration
_LAZY_IMPORTS = {
    # Core models
    "SummaryModel": ("kura.summarisation", "SummaryModel"),
    "ClusterDescriptionModel": ("kura.cluster", "ClusterDescriptionModel"),
    "MetaClusterModel": ("kura.meta_cluster", "MetaClusterModel"),
    
    # Types
    "Conversation": ("kura.types", "Conversation"),
    
    # Checkpoint managers
    "CheckpointManager": ("kura.checkpoint", "CheckpointManager"),
    "MultiCheckpointManager": ("kura.checkpoints", "MultiCheckpointManager"),
    
    # Clustering methods
    "KmeansClusteringMethod": ("kura.k_means", "KmeansClusteringMethod"),
    "MiniBatchKmeansClusteringMethod": ("kura.k_means", "MiniBatchKmeansClusteringMethod"),
    "HDBSCANClusteringMethod": ("kura.hdbscan", "HDBSCANClusteringMethod"),
    
    # Procedural functions
    "summarise_conversations": ("kura.summarisation", "summarise_conversations"),
    "generate_base_clusters_from_conversation_summaries": ("kura.cluster", "generate_base_clusters_from_conversation_summaries"),
    "reduce_clusters_from_base_clusters": ("kura.v1.kura", "reduce_clusters_from_base_clusters"),
    "reduce_dimensionality_from_clusters": ("kura.v1.kura", "reduce_dimensionality_from_clusters"),
    
    # Visualization
    "visualise_pipeline_results": ("kura.v1.visualization", "visualise_pipeline_results"),
    "visualise_clusters_rich": ("kura.v1.visualization", "visualise_clusters_rich"),
    "visualise_clusters_enhanced": ("kura.v1.visualization", "visualise_clusters_enhanced"),
    "visualise_clusters": ("kura.v1.visualization", "visualise_clusters"),
}

# Optional imports that may not be available
_OPTIONAL_IMPORTS = {
    "ParquetCheckpointManager": ("kura.checkpoints.parquet", "ParquetCheckpointManager"),
    "HFDatasetCheckpointManager": ("kura.checkpoints.hf_dataset", "HFDatasetCheckpointManager"),
}

def __getattr__(name: str):
    """Lazy loading of modules using __getattr__."""
    # Check if already cached
    if name in globals():
        return globals()[name]
    
    # Check required imports
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        try:
            from importlib import import_module
            module = import_module(module_path)
            attr = getattr(module, attr_name)
            globals()[name] = attr  # Cache it
            return attr
        except ImportError as e:
            raise ImportError(f"Failed to import {name} from {module_path}: {e}")
    
    # Check optional imports
    if name in _OPTIONAL_IMPORTS:
        module_path, attr_name = _OPTIONAL_IMPORTS[name]
        try:
            from importlib import import_module
            module = import_module(module_path)
            attr = getattr(module, attr_name)
            globals()[name] = attr  # Cache it
            return attr
        except ImportError:
            raise ImportError(f"Optional dependency {name} is not available. Install the required packages.")
    
    raise AttributeError(f"module 'kura' has no attribute '{name}'")

__all__ = list(_LAZY_IMPORTS.keys()) + list(_OPTIONAL_IMPORTS.keys())

__version__ = "1.0.0"
