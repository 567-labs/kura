"""
Cache strategy implementations for intermediate processing checkpoints.

This module provides caching strategies for job resumption and intermediate
checkpoints during processing.
"""

import os
from abc import ABC, abstractmethod
from typing import Any, Optional
import diskcache


class CacheStrategy(ABC):
    """Abstract base class for caching strategies."""

    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Retrieve a value from the cache by key."""
        raise NotImplementedError("Subclasses must implement get method")

    @abstractmethod
    def set(self, key: str, value: Any) -> None:
        """Store a value in the cache with the given key."""
        raise NotImplementedError("Subclasses must implement set method")


class DiskCacheStrategy(CacheStrategy):
    """Disk-based cache strategy using diskcache."""
    
    def __init__(self, cache_dir: str):
        """Initialize disk cache strategy.
        
        Args:
            cache_dir: Directory to store cache files
        """
        os.makedirs(cache_dir, exist_ok=True)
        self.cache = diskcache.Cache(cache_dir)
    
    def get(self, key: str) -> Optional[Any]:
        """Retrieve a value from the disk cache."""
        return self.cache.get(key)
    
    def set(self, key: str, value: Any) -> None:
        """Store a value in the disk cache."""
        self.cache[key] = value
