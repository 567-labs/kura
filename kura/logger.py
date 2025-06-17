import logging
from typing import Any, Dict, Optional, Union

from kura.base_classes import BaseClusterLogger


class StandardLogger(BaseClusterLogger):
    """Default logger implementation using Python's standard logging."""

    def __init__(self, name: str = "kura_clustering"):
        """Initialize the standard logger."""
        self.name = name
        self.logger = logging.getLogger(name)

        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    # Standard logging methods (required by base class)
    def info(self, message: str, **kwargs) -> None:
        """Log an info message."""
        self.logger.info(message, **kwargs)

    def debug(self, message: str, **kwargs) -> None:
        """Log a debug message."""
        self.logger.debug(message, **kwargs)

    def warning(self, message: str, **kwargs) -> None:
        """Log a warning message."""
        self.logger.warning(message, **kwargs)

    def error(self, message: str, exc_info: bool = False, **kwargs) -> None:
        """Log an error message with optional exception info."""
        self.logger.error(message, exc_info=exc_info, **kwargs)

    # Clustering-specific methods (required by base class)
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

    def log_artifact(self, file_path: str, name: Optional[str] = None, **metadata) -> None:
        """Log file artifacts (StandardLogger does nothing with artifacts)."""
        pass
