from abc import ABC, abstractmethod
from typing import Optional
import os
import hashlib
import diskcache


class BaseEmbeddingModel(ABC):
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize embedding model with optional disk caching.
        
        Args:
            cache_dir: Directory for disk cache storage (optional, defaults to no caching)
        """
        # Initialize disk cache only if cache_dir is provided
        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)
            self.cache = diskcache.Cache(cache_dir)
        else:
            self.cache = None

    def _get_cache_key(self, text: str, **kwargs) -> str:
        """
        Generate cache key from text and model parameters.
        
        Override this method in subclasses if custom caching behavior is needed.
        
        Args:
            text: The text to embed
            **kwargs: Additional parameters that affect embedding output
            
        Returns:
            MD5 hash string to use as cache key
        """
        # Get model name from subclass, fallback to class name if not available
        model_name = getattr(self, 'model_name', self.__class__.__name__)
        cache_components = (
            text,
            model_name,
            tuple(sorted(kwargs.items()))  # Any additional parameters
        )
        return hashlib.md5(str(cache_components).encode()).hexdigest()

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
