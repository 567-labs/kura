"""
Cached Embedding Model implementation.

This module provides a caching wrapper for embedding models that stores
embeddings using CacheStrategy for intermediate processing checkpoints.
"""

import hashlib
import json
import logging
from typing import Any, Dict, List, Optional

from kura.base_classes import BaseEmbeddingModel
from kura.cache_strategy import CacheStrategy

logger = logging.getLogger(__name__)


class CachedEmbeddingModel(BaseEmbeddingModel):
    """Embedding model with caching support for job resumption.
    
    This class wraps any BaseEmbeddingModel and adds caching functionality
    using a CacheStrategy. It caches embeddings based on model name,
    text content, and additional kwargs.
    """

    def __init__(
        self,
        embedding_model: BaseEmbeddingModel,
        cache_strategy: Optional[CacheStrategy] = None,
        **kwargs: Any
    ):
        """Initialize cached embedding model.

        Args:
            embedding_model: The underlying embedding model to wrap
            cache_strategy: Optional cache strategy for storing embeddings
            **kwargs: Additional parameters that should be included in cache keys
        """
        self.embedding_model = embedding_model
        self.cache_strategy = cache_strategy
        self.cache_kwargs = kwargs
        
        # Ensure the embedding model has a model_name attribute
        if not hasattr(self.embedding_model, 'model_name'):
            raise ValueError(
                f"Embedding model {type(self.embedding_model).__name__} must have a 'model_name' attribute"
            )
        
        logger.info(
            f"Initialized CachedEmbeddingModel with model={self.embedding_model.model_name}, "
            f"caching={'enabled' if cache_strategy else 'disabled'}"
        )

    def slug(self) -> str:
        """Return a unique identifier for the cached embedding model."""
        return self.embedding_model.slug()

    def _generate_cache_key(self, text: str) -> str:
        """Generate a cache key for a given text.
        
        Args:
            text: The text to generate a cache key for
            
        Returns:
            Unique cache key string
        """
        # Create a dictionary with all cache-relevant parameters
        cache_data = {
            "model_name": self.embedding_model.model_name,
            "text": text,
            **self.cache_kwargs
        }
        
        # Sort keys for consistent hashing
        cache_str = json.dumps(cache_data, sort_keys=True)
        
        # Generate hash
        return hashlib.sha256(cache_str.encode()).hexdigest()

    async def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed texts with caching support.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embeddings in the same order as input texts
        """
        if not texts:
            return []
        
        # If caching is disabled, just use the underlying model
        if not self.cache_strategy:
            return await self.embedding_model.embed(texts)
        
        logger.debug(f"Processing {len(texts)} texts with caching enabled")
        
        # Check cache for each text and collect results
        cached_embeddings: Dict[str, List[float]] = {}
        uncached_texts = []
        
        for text in texts:
            cache_key = self._generate_cache_key(text)
            cached_embedding = self.cache_strategy.get(cache_key)
            
            if cached_embedding is not None:
                cached_embeddings[text] = cached_embedding
                logger.debug(f"Cache hit for text: {text[:50]}...")
            else:
                uncached_texts.append(text)
        
        logger.debug(f"Found {len(cached_embeddings)} cached embeddings, {len(uncached_texts)} need embedding")
        
        # Embed uncached texts
        new_embeddings = {}
        if uncached_texts:
            fresh_embeddings = await self.embedding_model.embed(uncached_texts)
            
            # Cache new embeddings
            for text, embedding in zip(uncached_texts, fresh_embeddings):
                cache_key = self._generate_cache_key(text)
                self.cache_strategy.set(cache_key, embedding)
                new_embeddings[text] = embedding
                logger.debug(f"Cached embedding for text: {text[:50]}...")
        
        # Return embeddings in original order
        result_embeddings = []
        for text in texts:
            if text in cached_embeddings:
                result_embeddings.append(cached_embeddings[text])
            elif text in new_embeddings:
                result_embeddings.append(new_embeddings[text])
            else:
                raise RuntimeError(f"No embedding found for text: {text[:50]}...")
        
        return result_embeddings
