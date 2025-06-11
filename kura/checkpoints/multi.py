"""
Multi-checkpoint manager for coordinating multiple checkpoint storage backends.

This module provides the MultiCheckpointManager class that enables:
- Redundant storage across multiple backends
- Performance optimization through strategic loading/saving
- Environment-specific checkpoint management
- Flexible save and load strategies
"""

import logging
from typing import List, Optional, TypeVar, Literal
from pydantic import BaseModel

from kura.base_classes import BaseCheckpointManager

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class MultiCheckpointManager(BaseCheckpointManager):
    """Manages multiple checkpoint managers for redundancy, performance, or separation.

    This class coordinates multiple checkpoint managers, allowing you to:
    - Save to multiple backends for redundancy (e.g., local + cloud backup)
    - Load from the fastest available source
    - Separate environments (dev/staging/prod)
    - Implement different retention policies

    Attributes:
        managers: List of checkpoint managers to coordinate
        save_strategy: How to save checkpoints:
            - "all_enabled": Save to all enabled managers (default)
            - "primary_only": Save only to the first enabled manager
        load_strategy: How to load checkpoints:
            - "first_found": Return first successful load (default)
            - "priority": Try managers in order, return first success

    Example:
        >>> from kura.checkpoint import CheckpointManager
        >>> local_mgr = CheckpointManager("./local_checkpoints")
        >>> cloud_mgr = CheckpointManager("./cloud_backup")
        >>> multi_mgr = MultiCheckpointManager(
        ...     managers=[local_mgr, cloud_mgr],
        ...     save_strategy="all_enabled",
        ...     load_strategy="first_found"
        ... )
    """

    def __init__(
        self,
        managers: List[BaseCheckpointManager],
        *,
        save_strategy: Literal["all_enabled", "primary_only"] = "all_enabled",
        load_strategy: Literal["first_found", "priority"] = "first_found",
    ):
        """Initialize multi-checkpoint manager.

        Args:
            managers: List of checkpoint managers to coordinate
            save_strategy: Strategy for saving checkpoints
            load_strategy: Strategy for loading checkpoints
        """
        if not managers:
            raise ValueError("At least one checkpoint manager must be provided")

        self.managers = managers
        self.save_strategy = save_strategy
        self.load_strategy = load_strategy

        # Consider enabled if any manager is enabled
        enabled = any(mgr.enabled for mgr in managers)

        # Initialize base class with dummy checkpoint_dir
        super().__init__("", enabled=enabled)

        logger.info(
            f"Initialized MultiCheckpointManager with {len(managers)} managers "
            f"(save: {save_strategy}, load: {load_strategy})"
        )

    def setup_checkpoint_dir(self) -> None:
        """Setup checkpoint directories for all managers."""
        for mgr in self.managers:
            if mgr.enabled:
                mgr.setup_checkpoint_dir()

    def load_checkpoint(
        self, filename: str, model_class: type[T], **kwargs
    ) -> Optional[List[T]]:
        """Load checkpoint from first available manager.

        Args:
            filename: Name of checkpoint file
            model_class: Pydantic model class for deserializing
            **kwargs: Additional arguments passed to managers

        Returns:
            Loaded data if found, None otherwise
        """
        if not self.enabled:
            return None

        if self.load_strategy == "first_found":
            # Try all managers, return first successful load
            for mgr in self.managers:
                if mgr.enabled:
                    try:
                        data = mgr.load_checkpoint(filename, model_class, **kwargs)
                        if data is not None:
                            logger.info(
                                f"Loaded {filename} from {mgr.__class__.__name__}"
                            )
                            return data
                    except Exception as e:
                        logger.warning(
                            f"Failed to load {filename} from {mgr.__class__.__name__}: {e}"
                        )
                        continue

        elif self.load_strategy == "priority":
            # Try managers in order, return first success
            for mgr in self.managers:
                if mgr.enabled:
                    try:
                        data = mgr.load_checkpoint(filename, model_class, **kwargs)
                        if data is not None:
                            logger.info(
                                f"Loaded {filename} from {mgr.__class__.__name__} (priority)"
                            )
                            return data
                    except Exception as e:
                        logger.warning(
                            f"Failed to load {filename} from {mgr.__class__.__name__}: {e}"
                        )
                        # In priority mode, don't continue to next manager on error
                        break

        logger.info(f"No checkpoint found for {filename} in any manager")
        return None

    def save_checkpoint(self, filename: str, data: List[T], **kwargs) -> None:
        """Save checkpoint according to configured strategy.

        Args:
            filename: Name of checkpoint file
            data: Data to save
            **kwargs: Additional arguments passed to managers
        """
        if not self.enabled:
            return

        if self.save_strategy == "all_enabled":
            # Save to all enabled managers
            saved_count = 0
            for mgr in self.managers:
                if mgr.enabled:
                    try:
                        mgr.save_checkpoint(filename, data, **kwargs)
                        saved_count += 1
                        logger.debug(f"Saved {filename} to {mgr.__class__.__name__}")
                    except Exception as e:
                        logger.warning(
                            f"Failed to save {filename} to {mgr.__class__.__name__}: {e}"
                        )

            if saved_count > 0:
                logger.info(f"Saved {filename} to {saved_count} checkpoint manager(s)")
            else:
                logger.error(f"Failed to save {filename} to any checkpoint manager")

        elif self.save_strategy == "primary_only":
            # Save only to first enabled manager
            for mgr in self.managers:
                if mgr.enabled:
                    try:
                        mgr.save_checkpoint(filename, data, **kwargs)
                        logger.info(
                            f"Saved {filename} to primary manager {mgr.__class__.__name__}"
                        )
                        return
                    except Exception as e:
                        logger.error(
                            f"Failed to save {filename} to primary manager: {e}"
                        )
                        raise

            logger.warning("No enabled checkpoint managers available for saving")

    def list_checkpoints(self) -> List[str]:
        """List all unique checkpoints across all managers.

        Returns:
            Combined list of unique checkpoint filenames
        """
        all_checkpoints = set()

        for mgr in self.managers:
            if mgr.enabled:
                try:
                    checkpoints = mgr.list_checkpoints()
                    all_checkpoints.update(checkpoints)
                except Exception as e:
                    logger.warning(
                        f"Failed to list checkpoints from {mgr.__class__.__name__}: {e}"
                    )

        return sorted(list(all_checkpoints))

    def delete_checkpoint(self, filename: str) -> bool:
        """Delete checkpoint from all managers.

        Args:
            filename: Name of checkpoint file to delete

        Returns:
            True if deleted from at least one manager
        """
        if not self.enabled:
            return False

        deleted = False
        for mgr in self.managers:
            if mgr.enabled:
                try:
                    if mgr.delete_checkpoint(filename):
                        deleted = True
                        logger.debug(
                            f"Deleted {filename} from {mgr.__class__.__name__}"
                        )
                except Exception as e:
                    logger.warning(
                        f"Failed to delete {filename} from {mgr.__class__.__name__}: {e}"
                    )

        if deleted:
            logger.info(f"Deleted {filename} from checkpoint manager(s)")

        return deleted

    def __repr__(self) -> str:
        """String representation of multi-checkpoint manager."""
        manager_names = [mgr.__class__.__name__ for mgr in self.managers]
        return (
            f"MultiCheckpointManager(managers={manager_names}, "
            f"save_strategy='{self.save_strategy}', "
            f"load_strategy='{self.load_strategy}', "
            f"enabled={self.enabled})"
        )
