from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kura.types import Cluster, ProjectedCluster


class BaseDimensionalityReduction(ABC):
    @property
    @abstractmethod
    def checkpoint_filename(self) -> str:
        """The filename to use for checkpointing this model's output."""
        pass

    @abstractmethod
    async def reduce_dimensionality(
        self, clusters: list["Cluster"]
    ) -> list["ProjectedCluster"]:
        """
        This reduces the dimensionality of the individual clusters that we've created so we can visualise them in a lower dimension
        """
        pass
