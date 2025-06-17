from abc import ABC, abstractmethod
from typing import Any, Optional
import os
import diskcache


class CacheStrategy(ABC):
    """Abstract base class for caching strategies."""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Retrieve a value from the cache by key."""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any) -> None:
        """Store a value in the cache with the given key."""
        pass


class DiskCacheStrategy(CacheStrategy):
    """Disk-based caching strategy using diskcache."""
    
    def __init__(self, cache_dir: str):
        """
        Initialize disk cache strategy.
        
        Args:
            cache_dir: Directory path for cache storage
        """
        os.makedirs(cache_dir, exist_ok=True)
        self.cache = diskcache.Cache(cache_dir)
    
    def get(self, key: str) -> Optional[Any]:
        """Retrieve a value from the disk cache."""
        return self.cache.get(key)
    
    def set(self, key: str, value: Any) -> None:
        """Store a value in the disk cache."""
        self.cache.set(key, value)


class NoCacheStrategy(CacheStrategy):
    """No-op caching strategy that doesn't cache anything."""
    
    def get(self, key: str) -> Optional[Any]:
        """Always returns None (no cache hit)."""
        return None
    
    def set(self, key: str, value: Any) -> None:
        """Does nothing (no caching)."""
        pass
