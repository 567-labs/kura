"""
Base class for checkpoint managers.

This module defines the abstract base class that all checkpoint managers
should inherit from, ensuring a consistent interface across different
checkpoint implementations.
"""

from abc import ABC, abstractmethod
from typing import Optional, TypeVar, List
from pathlib import Path
import logging

from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class BaseCheckpointManager(ABC):
    """Abstract base class for checkpoint managers.

    This class defines the common interface that all checkpoint managers
    must implement, ensuring consistent behavior across different storage
    backends (JSONL, Parquet, HuggingFace datasets, etc.).
    """

    def __init__(self, checkpoint_dir: str, *, enabled: bool = True):
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory for saving checkpoints
            enabled: Whether checkpointing is enabled
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.enabled = enabled

        if self.enabled:
            self.setup_checkpoint_dir()

    @abstractmethod
    def setup_checkpoint_dir(self) -> None:
        """Create checkpoint directory if it doesn't exist.

        This method should be implemented by subclasses to handle
        any setup required for the checkpoint storage backend.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def load_checkpoint(
        self, filename: str, model_class: type[T], **kwargs
    ) -> Optional[List[T]]:
        """Load data from a checkpoint file if it exists.

        Args:
            filename: Name of the checkpoint file
            model_class: Pydantic model class for deserializing the data
            **kwargs: Additional arguments specific to the implementation

        Returns:
            List of model instances if checkpoint exists, None otherwise
        """
        pass

    @abstractmethod
    def save_checkpoint(self, filename: str, data: List[T], **kwargs) -> None:
        """Save data to a checkpoint file.

        Args:
            filename: Name of the checkpoint file
            data: List of model instances to save
            **kwargs: Additional arguments specific to the implementation
        """
        pass

    @abstractmethod
    def list_checkpoints(self) -> List[str]:
        """List all available checkpoint files.

        Returns:
            List of checkpoint filenames available in the storage backend
        """
        pass

    def delete_checkpoint(self, filename: str) -> bool:
        """Delete a checkpoint file.

        Args:
            filename: Name of the checkpoint file to delete

        Returns:
            True if file was deleted, False if it didn't exist

        Note:
            This method provides a default implementation that can be
            overridden by subclasses for backend-specific deletion logic.
        """
        if not self.enabled:
            return False

        # Default implementation - subclasses should override this
        logger.warning(
            f"delete_checkpoint not implemented for {self.__class__.__name__}"
        )
        return False

    def get_checkpoint_path(self, filename: str) -> Path:
        """Get full path for a checkpoint file.

        Args:
            filename: Name of the checkpoint file

        Returns:
            Path object for the checkpoint file

        Note:
            This is a convenience method that subclasses can override
            if they need different path handling logic.
        """
        return self.checkpoint_dir / filename

    def is_enabled(self) -> bool:
        """Check if checkpointing is enabled.

        Returns:
            True if checkpointing is enabled, False otherwise
        """
        return self.enabled

    def __repr__(self) -> str:
        """String representation of the checkpoint manager."""
        return f"{self.__class__.__name__}(checkpoint_dir='{self.checkpoint_dir}', enabled={self.enabled})"
