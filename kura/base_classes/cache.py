from abc import ABC, abstractmethod
from typing import Any, Optional


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
