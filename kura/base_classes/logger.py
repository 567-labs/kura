from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union


class BaseClusterLogger(ABC):
    """
    Abstract base class for logging clustering experiments and analysis.

    This interface provides a unified way to log clustering parameters, metrics,
    errors, and artifacts across different logging providers.
    """

    @abstractmethod
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log clustering parameters and configuration."""
        raise NotImplementedError

    @abstractmethod
    def log_metrics(
        self, metrics: Dict[str, Union[float, int]], step: Optional[int] = None
    ) -> None:
        """Log clustering metrics and performance stats."""
        raise NotImplementedError

    @abstractmethod
    def log_errors(
        self, error: Union[str, Exception], context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log errors with optional context for debugging."""
        raise NotImplementedError

    @abstractmethod
    def log(self, data: Any, key: str, **metadata) -> None:
        """Generic logging method for arbitrary data."""
        raise NotImplementedError

    @property
    @abstractmethod
    def checkpoint_filename(self) -> str:
        """Return the filename to use for checkpointing this logger's configuration."""
        raise NotImplementedError
