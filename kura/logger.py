import logging

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


def create_logger(name: str = "kura_clustering") -> BaseClusterLogger:
    """Create a standard logger instance."""
    return StandardLogger(name=name)
