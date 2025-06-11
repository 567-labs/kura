from abc import ABC, abstractmethod
from kura.types.summarisation import ConversationSummary
from kura.types.cluster import Cluster
from typing import Dict, List


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
        pass
