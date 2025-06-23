from abc import ABC, abstractmethod
from typing import Union, TYPE_CHECKING

if TYPE_CHECKING:
    from kura.types.summarisation import ConversationSummary


class BaseClusteringMethod(ABC):
    @abstractmethod
    def cluster(
        self, items: list[dict[str, Union["ConversationSummary", list[float]]]]
    ) -> dict[int, list["ConversationSummary"]]:
        """Clustering method takes in a item + embedding and returns a mapping of cluster ids to items"""
        pass
