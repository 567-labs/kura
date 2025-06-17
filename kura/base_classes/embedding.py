from abc import ABC, abstractmethod
from typing import Optional, Callable, Any
import os
import hashlib
import diskcache
import logging

logger = logging.getLogger(__name__)


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

    async def _embed_with_cache(
        self, 
        texts: list[str], 
        embed_fn: Callable[[list[str]], Any],
        cache_key_kwargs: dict[str, Any] = None
    ) -> list[list[float]]:
        """
        Handle caching logic for embedding operations.
        
        Args:
            texts: List of texts to embed
            embed_fn: Function to call for uncached texts (should be async)
            cache_key_kwargs: Additional kwargs for cache key generation
            
        Returns:
            List of embeddings in the same order as input texts
        """
        if not texts:
            logger.debug("Empty text list provided, returning empty embeddings")
            return []

        # If caching is disabled, use the provided function directly
        if self.cache is None:
            return await embed_fn(texts)

        # Use empty dict if no cache key kwargs provided
        if cache_key_kwargs is None:
            cache_key_kwargs = {}

        # Check cache for each text
        cached_embeddings = {}
        uncached_texts = []
        uncached_indices = []

        for i, text in enumerate(texts):
            cache_key = self._get_cache_key(text, **cache_key_kwargs)
            cached_embedding = self.cache.get(cache_key)
            if cached_embedding is not None:
                cached_embeddings[i] = cached_embedding
                logger.debug(f"Found cached embedding for text at index {i}")
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)

        logger.info(f"Found {len(cached_embeddings)} cached embeddings, need to generate {len(uncached_texts)} new embeddings")

        # Generate embeddings for uncached texts
        new_embeddings = []
        if uncached_texts:
            new_embeddings = await embed_fn(uncached_texts)
            
            # Cache the new embeddings
            for text, embedding in zip(uncached_texts, new_embeddings):
                cache_key = self._get_cache_key(text, **cache_key_kwargs)
                self.cache.set(cache_key, embedding)
                logger.debug(f"Cached embedding for text: {text[:50]}...")

        # Combine cached and new embeddings in original order
        result = []
        new_embedding_idx = 0
        for i in range(len(texts)):
            if i in cached_embeddings:
                result.append(cached_embeddings[i])
            else:
                result.append(new_embeddings[new_embedding_idx])
                new_embedding_idx += 1

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
