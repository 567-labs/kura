from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union


class BaseClusterLogger(ABC):
    """
    Abstract base class for logging clustering experiments and analysis.
    
    This interface provides standard logging methods and clustering-specific
    convenience methods across different logging providers.
    """

    # Standard logging methods (must be implemented)
    @abstractmethod
    def info(self, message: str, **kwargs) -> None:
        """Log an info message."""
        raise NotImplementedError

    @abstractmethod
    def debug(self, message: str, **kwargs) -> None:
        """Log a debug message."""
        raise NotImplementedError

    @abstractmethod
    def warning(self, message: str, **kwargs) -> None:
        """Log a warning message."""
        raise NotImplementedError

    @abstractmethod
    def error(self, message: str, exc_info: bool = False, **kwargs) -> None:
        """Log an error message with optional exception info."""
        raise NotImplementedError

    # Clustering-specific convenience methods (default implementations)
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log clustering parameters and configuration."""
        self.info(f"PARAMS: {params}")

    def log_metrics(self, metrics: Dict[str, Union[float, int]], step: Optional[int] = None) -> None:
        """Log clustering metrics and performance stats."""
        step_info = f" (step {step})" if step is not None else ""
        self.info(f"METRICS{step_info}: {metrics}")

    def log_errors(self, error: Union[str, Exception], context: Optional[Dict[str, Any]] = None) -> None:
        """Log errors with optional context for debugging."""
        if isinstance(error, Exception):
            if context:
                self.error(f"ERROR with context {context}: {error}", exc_info=True)
            else:
                self.error(f"ERROR: {error}", exc_info=True)
        else:
            if context:
                self.error(f"ERROR: {error} | Context: {context}")
            else:
                self.error(f"ERROR: {error}")

    def log(self, data: Any, key: str, **metadata) -> None:
        """Generic logging method for arbitrary data."""
        metadata_str = f" | Metadata: {metadata}" if metadata else ""
        self.info(f"LOG [{key}]: {data}{metadata_str}")
