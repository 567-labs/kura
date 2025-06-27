
from pathlib import Path
from typing import Optional, List, TypeVar
import logging
from pydantic import BaseModel

from kura.base_classes import BaseCheckpointManager

T = TypeVar("T", bound=BaseModel)

logger = logging.getLogger(__name__)


class CheckpointManager(BaseCheckpointManager):
    """Handles checkpoint loading and saving for pipeline steps."""

    def __init__(self, checkpoint_dir: str, *, enabled: bool = True):
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory for saving checkpoints
            enabled: Whether checkpointing is enabled
        """
        super().__init__(checkpoint_dir, enabled=enabled)

    def setup_checkpoint_dir(self) -> None:
        """Create checkpoint directory if it doesn't exist."""
        if not self.checkpoint_dir.exists():
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created checkpoint directory: {self.checkpoint_dir}")

    def get_checkpoint_path(self, filename: str) -> Path:
        """Get full path for a checkpoint file.

        Args:
            filename: Name of the checkpoint file

        Returns:
            Path object for the checkpoint file
        """
        return self.checkpoint_dir / filename

    def load_checkpoint(self, filename: str, model_class: type[T], **kwargs) -> Optional[List[T]]:
        """Load data from a checkpoint file if it exists.

        Args:
            filename: Name of the checkpoint file
            model_class: Pydantic model class for deserializing the data
            **kwargs: Additional arguments (for compatibility with base class)

        Returns:
            List of model instances if checkpoint exists, None otherwise
        """
        if not self.enabled:
            return None

        checkpoint_path = self.get_checkpoint_path(filename)
        if checkpoint_path.exists():
            logger.info(
                f"Loading checkpoint from {checkpoint_path} for {model_class.__name__}"
            )
            with open(checkpoint_path, "r") as f:
                return [model_class.model_validate_json(line) for line in f]
        return None

    def save_checkpoint(self, filename: str, data: List[T], **kwargs) -> None:
        """Save data to a checkpoint file.

        Args:
            filename: Name of the checkpoint file
            data: List of model instances to save
            **kwargs: Additional arguments (for compatibility with base class)
        """
        if not self.enabled:
            return

        checkpoint_path = self.get_checkpoint_path(filename)
        with open(checkpoint_path, "w") as f:
            for item in data:
                f.write(item.model_dump_json() + "\n")
        logger.info(f"Saved checkpoint to {checkpoint_path} with {len(data)} items")

    def list_checkpoints(self) -> List[str]:
        """List all available checkpoint files."""
        if not self.enabled or not self.checkpoint_dir.exists():
            return []
        return [
            f.name
            for f in self.checkpoint_dir.iterdir()
            if f.is_file()
        ]

    def delete_checkpoint(self, filename: str) -> bool:
        """Delete a checkpoint file."""
        if not self.enabled:
            return False
        checkpoint_path = self.get_checkpoint_path(filename)
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            logger.info(f"Deleted checkpoint: {checkpoint_path}")
            return True
        return False
