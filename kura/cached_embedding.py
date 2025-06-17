"""
Cached Embedding Model implementation.

This module provides a caching wrapper for embedding models that stores
embeddings using configurable cache keys based on model name, text content,
and additional parameters.
"""

import hashlib
import json
import logging
from typing import Any, Dict, List, Optional

from kura.base_classes import BaseEmbeddingModel, BaseCheckpointManager

logger = logging.getLogger(__name__)


class CachedEmbeddingModel(BaseEmbeddingModel):
    """Embedding model with caching support.
    
    This class wraps any BaseEmbeddingModel and adds caching functionality
    using a BaseCheckpointManager. It caches embeddings based on model name,
    text content, and additional kwargs to avoid recomputing embeddings
    for the same inputs.
    """

    def __init__(
        self,
        embedding_model: BaseEmbeddingModel,
        cache_manager: Optional[BaseCheckpointManager] = None,
        **kwargs: Any
    ):
        """Initialize cached embedding model.

        Args:
            embedding_model: The underlying embedding model to wrap
            cache_manager: Optional cache manager for storing embeddings
            **kwargs: Additional parameters that should be included in cache keys
        """
        self.embedding_model = embedding_model
        self.cache_manager = cache_manager
        self.cache_kwargs = kwargs
        
        # Ensure the embedding model has a model_name attribute
        if not hasattr(self.embedding_model, 'model_name'):
            raise ValueError(
                f"Embedding model {type(self.embedding_model).__name__} must have a 'model_name' attribute"
            )
        
        logger.info(
            f"Initialized CachedEmbeddingModel with model={self.embedding_model.model_name}, "
            f"caching={'enabled' if cache_manager else 'disabled'}"
        )

    def slug(self) -> str:
        """Return a unique identifier for the cached embedding model."""
        base_slug = self.embedding_model.slug()
        if self.cache_manager:
            return f"cached:{base_slug}"
        return base_slug

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

    def _load_cached_embeddings(self, texts: List[str]) -> Dict[str, List[float]]:
        """Load cached embeddings for given texts.
        
        Args:
            texts: List of texts to check for cached embeddings
            
        Returns:
            Dictionary mapping cache keys to embeddings
        """
        if not self.cache_manager or not self.cache_manager.enabled:
            return {}
        
        cached_embeddings = {}
        
        for text in texts:
            cache_key = self._generate_cache_key(text)
            try:
                # Try to load individual embedding from cache
                # For now, we'll use a simple approach where each embedding is stored separately
                cached_data = self.cache_manager.load_checkpoint(
                    f"embedding_{cache_key}.json",
                    dict  # We'll store as dict for now
                )
                
                if cached_data and len(cached_data) > 0:
                    cached_embeddings[cache_key] = cached_data[0].get("embedding")
                    logger.debug(f"Loaded cached embedding for key: {cache_key[:8]}...")
                    
            except Exception as e:
                logger.debug(f"Failed to load cached embedding for key {cache_key[:8]}...: {e}")
                continue
        
        return cached_embeddings

    def _save_cached_embeddings(self, text_embedding_pairs: List[tuple[str, List[float]]]) -> None:
        """Save embeddings to cache.
        
        Args:
            text_embedding_pairs: List of (text, embedding) tuples to cache
        """
        if not self.cache_manager or not self.cache_manager.enabled:
            return
        
        for text, embedding in text_embedding_pairs:
            cache_key = self._generate_cache_key(text)
            try:
                # Save individual embedding
                self.cache_manager.save_checkpoint(
                    f"embedding_{cache_key}.json",
                    [{"text": text, "embedding": embedding, "cache_key": cache_key}]
                )
                logger.debug(f"Saved cached embedding for key: {cache_key[:8]}...")
                
            except Exception as e:
                logger.warning(f"Failed to save cached embedding for key {cache_key[:8]}...: {e}")

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
        if not self.cache_manager or not self.cache_manager.enabled:
            logger.debug("Caching disabled, using underlying model directly")
            return await self.embedding_model.embed(texts)
        
        logger.info(f"Processing {len(texts)} texts with caching enabled")
        
        # Generate cache keys for all texts
        text_to_cache_key = {text: self._generate_cache_key(text) for text in texts}
        
        # Load cached embeddings
        cached_embeddings = self._load_cached_embeddings(texts)
        
        # Identify uncached texts that need to be embedded
        uncached_texts = []
        for text in texts:
            cache_key = text_to_cache_key[text]
            if cache_key not in cached_embeddings:
                uncached_texts.append(text)
        
        logger.info(f"Found {len(cached_embeddings)} cached embeddings, {len(uncached_texts)} need embedding")
        
        # Embed uncached texts
        new_embeddings = {}
        if uncached_texts:
            logger.info(f"Embedding {len(uncached_texts)} uncached texts")
            fresh_embeddings = await self.embedding_model.embed(uncached_texts)
            
            # Map new embeddings to their cache keys
            text_embedding_pairs = []
            for text, embedding in zip(uncached_texts, fresh_embeddings):
                cache_key = text_to_cache_key[text]
                new_embeddings[cache_key] = embedding
                text_embedding_pairs.append((text, embedding))
            
            # Save new embeddings to cache
            self._save_cached_embeddings(text_embedding_pairs)
        
        # Combine cached and new embeddings in original order
        result_embeddings = []
        for text in texts:
            cache_key = text_to_cache_key[text]
            if cache_key in cached_embeddings:
                result_embeddings.append(cached_embeddings[cache_key])
            elif cache_key in new_embeddings:
                result_embeddings.append(new_embeddings[cache_key])
            else:
                raise RuntimeError(f"No embedding found for text: {text[:50]}...")
        
        logger.info(f"Successfully returned {len(result_embeddings)} embeddings")
        return result_embeddings
