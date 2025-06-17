from abc import ABC, abstractmethod
from typing import Optional
import os
import hashlib
import diskcache
import logging

logger = logging.getLogger(__name__)


class BaseEmbeddingModel(ABC):
    model_name: str  # Subclasses must set this
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize embedding model with optional disk caching.
        
        Args:
            cache_dir: Directory for disk cache storage (optional, defaults to no caching)
        """
        # Validate that subclass has set model_name
        if not hasattr(self, 'model_name') or not self.model_name:
            raise AttributeError(f"{self.__class__.__name__} must set self.model_name")
            
        # Initialize disk cache only if cache_dir is provided
        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)
            self.cache = diskcache.Cache(cache_dir)
        else:
            self.cache = None

    def _get_cache_key(self, text: str, **kwargs) -> str:
        """
        Generate cache key from text and model parameters.
        
        Args:
            text: The text to embed
            **kwargs: Additional parameters that affect embedding output
            
        Returns:
            MD5 hash string to use as cache key
        """
        cache_components = (
            text,
            self.model_name,  # Assume this always exists
            tuple(sorted(kwargs.items()))  # Any additional parameters
        )
        return hashlib.md5(str(cache_components).encode()).hexdigest()

    async def _embed_with_cache(
        self, 
        texts: list[str], 
        embed_fn,
        **cache_kwargs
    ) -> list[list[float]]:
        """
        Handle caching logic for embedding operations.
        
        Args:
            texts: List of texts to embed
            embed_fn: Async function to call for uncached texts
            **cache_kwargs: Additional kwargs for cache key generation
            
        Returns:
            List of embeddings in the same order as input texts
        """
        if not texts:
            logger.debug("Empty text list provided, returning empty embeddings")
            return []

        # If caching is disabled, use the provided function directly
        if self.cache is None:
            return await embed_fn(texts)

        # Find uncached texts
        uncached_texts = []
        for text in texts:
            cache_key = self._get_cache_key(text, **cache_kwargs)
            if self.cache.get(cache_key) is None:
                uncached_texts.append(text)

        logger.info(f"Found {len(texts) - len(uncached_texts)} cached embeddings, need to generate {len(uncached_texts)} new embeddings")

        # Embed and cache uncached texts
        if uncached_texts:
            new_embeddings = await embed_fn(uncached_texts)
            for text, embedding in zip(uncached_texts, new_embeddings):
                cache_key = self._get_cache_key(text, **cache_kwargs)
                self.cache.set(cache_key, embedding)
                logger.debug(f"Cached embedding for text: {text[:50]}...")

        # Return all embeddings from cache in original order
        result = []
        for text in texts:
            cache_key = self._get_cache_key(text, **cache_kwargs)
            result.append(self.cache.get(cache_key))

        logger.info(f"Successfully embedded {len(texts)} texts, produced {len(result)} embeddings")
        return result

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
