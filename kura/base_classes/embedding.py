from abc import ABC, abstractmethod


class BaseEmbeddingModel(ABC):
    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts into a list of lists of floats"""
        pass

    @abstractmethod
    def slug(self) -> str:
        """Return a unique identifier for the embedding model.
        This is used to identify the embedding model in the checkpoint manager.
        """
        pass
