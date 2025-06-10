from .v1.kura import (
    summarise_conversations,
    generate_base_clusters_from_conversation_summaries,
    reduce_clusters_from_base_clusters,
    reduce_dimensionality_from_clusters,
    CheckpointManager,
)
from .cluster import ClusterModel
from .meta_cluster import MetaClusterModel
from .summarisation import SummaryModel
from .types import Conversation
from .k_means import KmeansClusteringMethod
from .hdbscan import HDBSCANClusteringMethod

# Optional VLLM imports (require vllm dependency group)
try:
    from .vllm_embedding import VLLMEmbeddingModel, create_vllm_embedding_model_for_scale
    from .vllm_summarisation import VLLMSummaryModel, create_vllm_summary_model_for_scale
    from .vllm_config import VLLMConfigManager, auto_select_models, should_use_vllm
    _VLLM_AVAILABLE = True
except ImportError:
    _VLLM_AVAILABLE = False

__all__ = [
    "ClusterModel",
    "MetaClusterModel",
    "SummaryModel",
    "Conversation",
    "KmeansClusteringMethod",
    "HDBSCANClusteringMethod",
    "summarise_conversations",
    "generate_base_clusters_from_conversation_summaries",
    "reduce_clusters_from_base_clusters",
    "reduce_dimensionality_from_clusters",
    "CheckpointManager",
]

# Add VLLM models to __all__ if available
if _VLLM_AVAILABLE:
    __all__.extend([
        "VLLMEmbeddingModel",
        "create_vllm_embedding_model_for_scale",
        "VLLMSummaryModel", 
        "create_vllm_summary_model_for_scale",
        "VLLMConfigManager",
        "auto_select_models",
        "should_use_vllm",
    ])
