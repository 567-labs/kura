from abc import ABC, abstractmethod
from kura.types.summarisation import ConversationSummary


class BaseClusteringMethod(ABC):
    @abstractmethod
    def cluster(
        self, items: list[ConversationSummary]
    ) -> dict[int, list[ConversationSummary]]:
        pass
