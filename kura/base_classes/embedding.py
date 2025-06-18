from abc import ABC, abstractmethod
import hashlib
import json


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

    def _get_cache_key_data(self, text: str) -> dict:
        """Override this to add model-specific cache key data."""
        return {
            "model_name": getattr(self, "model_name", self.__class__.__name__),
            "text": text,
        }

    def _generate_cache_key(self, text: str) -> str:
        """Generate a cache key for a given text."""
        cache_data = self._get_cache_key_data(text)
        cache_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.sha256(cache_str.encode()).hexdigest()
