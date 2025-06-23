# Light imports - these are fast to load
from .types import Conversation
import importlib as _importlib

# Submodules that can be lazy-loaded
_submodules = [
    "checkpoint",
    "checkpoints", 
    "k_means",
    "hdbscan",
    "cluster", 
    "summarisation",
    "embedding",
    "meta_cluster",
    "dimensionality",
    "v1",
]

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
    # Handle submodule access (e.g., kura.k_means)
    if name in _submodules:
        module = _importlib.import_module(f"kura.{name}")
        globals()[name] = module  # Cache for subsequent accesses
        return module
    
    # Auto-search for classes/functions in submodules (e.g., kura.KmeansClusteringMethod)
    for submodule_name in _submodules:
        try:
            module = _importlib.import_module(f"kura.{submodule_name}")
            if hasattr(module, name):
                attr = getattr(module, name)
                globals()[name] = attr  # Cache for subsequent accesses
                return attr
        except ImportError:
            continue
    
    raise AttributeError(f"module 'kura' has no attribute '{name}'")


# Auto-generate __all__ from submodules + known exports
__all__ = _submodules + [
    # Always available (not lazy-loaded)
    "Conversation",
    # Main exports from submodules (for better IDE support)
    "SummaryModel",
    "ClusterDescriptionModel", 
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
