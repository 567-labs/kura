from typing import Any, Optional
import os
import diskcache

from kura.base_classes.cache import CacheStrategy


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

