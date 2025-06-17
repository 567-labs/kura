from abc import ABC, abstractmethod
from kura.types.summarisation import ConversationSummary
from kura.types.cluster import Cluster
from typing import Dict, List, Optional
import hashlib


class BaseClusterDescriptionModel(ABC):
    @property
    @abstractmethod
    def checkpoint_filename(self) -> str:
        """The filename to use for checkpointing this model's output."""
        pass

    @abstractmethod
    async def generate_clusters(
        self,
        cluster_id_to_summaries: Dict[int, List[ConversationSummary]],
        prompt: str,
        max_contrastive_examples: int = 10,
    ) -> List[Cluster]:
        """Generate cluster descriptions for all clusters."""
        pass

    @abstractmethod
    async def generate_cluster_description(
        self,
        summaries: List[ConversationSummary],
        contrastive_examples: List[ConversationSummary],
        prompt: str,
        **kwargs
    ) -> "Cluster":
        """
        Generate a single cluster description with optional caching.
        
        Args:
            summaries: Summaries in this cluster  
            contrastive_examples: Examples from other clusters for contrast
            prompt: Prompt template for cluster generation
            **kwargs: Additional model parameters
            
        Returns:
            Generated cluster with name and description
        """
        pass
    
    def _get_cluster_cache_key(
        self,
        summaries: List[ConversationSummary],
        contrastive_examples: List[ConversationSummary],
        prompt: str,
        **kwargs
    ) -> str:
        """Generate cache key for a single cluster description."""
        # Hash cluster summaries (sorted for stability)
        cluster_reprs = [str(summary) for summary in summaries]
        cluster_hash = hashlib.md5(''.join(sorted(cluster_reprs)).encode()).hexdigest()
        
        # Hash contrastive examples (sorted for stability)
        contrastive_reprs = [str(summary) for summary in contrastive_examples]
        contrastive_hash = hashlib.md5(''.join(sorted(contrastive_reprs)).encode()).hexdigest()
        
        # Final cache key
        cache_components = (
            cluster_hash,
            contrastive_hash, 
            hashlib.md5(prompt.encode()).hexdigest(),
            self.checkpoint_filename,
        )
        
        return hashlib.md5(str(cache_components).encode()).hexdigest()
