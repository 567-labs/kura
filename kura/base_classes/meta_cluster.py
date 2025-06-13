from abc import ABC, abstractmethod
from kura.types.cluster import Cluster


class BaseMetaClusterModel(ABC):
    @property
    @abstractmethod
    def checkpoint_filename(self) -> str:
        """The filename to use for checkpointing this model's output."""
        pass

    @property
    @abstractmethod
    def max_clusters(self) -> int:
        """The maximum number of root clusters to reduce to."""
        pass

    @abstractmethod
    async def reduce_clusters(self, clusters: list[Cluster]) -> list[Cluster]:
        pass
