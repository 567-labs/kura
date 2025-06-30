"""
MultiCheckpointManager: Coordinate multiple checkpoint backends for redundancy and performance.

This module provides a checkpoint manager that can work with multiple backends simultaneously,
supporting different save and load strategies for flexibility and reliability.
"""

import logging
from pathlib import Path
from typing import List, Optional, TypeVar, Literal, Dict, Any
from pydantic import BaseModel

from kura.base_classes import BaseCheckpointManager

T = TypeVar("T", bound=BaseModel)
logger = logging.getLogger(__name__)


class MultiCheckpointManager(BaseCheckpointManager):
    """Manages multiple checkpoint backends with configurable strategies.

    This manager coordinates multiple checkpoint backends to provide:
    - Redundancy: Save to multiple backends for reliability
    - Performance: Load from the fastest available backend
    - Flexibility: Mix different checkpoint formats (JSONL, Parquet, HF Datasets)

    Example:
        >>> jsonl_mgr = CheckpointManager("./jsonl_checkpoints")
        >>> parquet_mgr = ParquetCheckpointManager("./parquet_checkpoints")
        >>>
        >>> multi_mgr = MultiCheckpointManager(
        ...     [jsonl_mgr, parquet_mgr],
        ...     save_strategy="all_enabled",  # Save to both
        ...     load_strategy="first_found"    # Load from first available
        ... )
    """

    def __init__(
        self,
        managers: List[BaseCheckpointManager],
        *,
        save_strategy: Literal["all_enabled", "primary_only"] = "all_enabled",
        load_strategy: Literal["first_found", "priority"] = "first_found",
    ):
        """Initialize MultiCheckpointManager.

        Args:
            managers: List of checkpoint managers to coordinate
            save_strategy: How to save checkpoints
                - "all_enabled": Save to all enabled managers
                - "primary_only": Save only to the first enabled manager
            load_strategy: How to load checkpoints
                - "first_found": Try each manager until one succeeds
                - "priority": Always use the first manager, fail if it doesn't have the checkpoint
        """
        if not managers:
            raise ValueError("At least one checkpoint manager must be provided")

        # Validate save strategy
        valid_save_strategies = {"all_enabled", "primary_only"}
        if save_strategy not in valid_save_strategies:
            raise ValueError(
                f"Invalid save_strategy '{save_strategy}'. Must be one of: {', '.join(sorted(valid_save_strategies))}"
            )

        # Validate load strategy
        valid_load_strategies = {"first_found", "priority"}
        if load_strategy not in valid_load_strategies:
            raise ValueError(
                f"Invalid load_strategy '{load_strategy}'. Must be one of: {', '.join(sorted(valid_load_strategies))}"
            )

        self.managers = managers
        self.save_strategy = save_strategy
        self.load_strategy = load_strategy

        # MultiCheckpointManager is enabled if any child manager is enabled
        self.enabled = any(mgr.enabled for mgr in self.managers)

        # Use the first manager's checkpoint directory as our base
        # (This is mainly for compatibility with the base class interface)
        self.checkpoint_dir = managers[0].checkpoint_dir

        logger.info(
            f"Initialized MultiCheckpointManager with {len(managers)} managers, "
            f"save_strategy={save_strategy}, load_strategy={load_strategy}"
        )

    def setup_checkpoint_dir(self) -> None:
        """Set up checkpoint directories for all enabled managers."""
        for mgr in self.managers:
            if mgr.enabled:
                mgr.setup_checkpoint_dir()

    def get_checkpoint_path(self, filename: str) -> Path:
        """Get checkpoint path from the primary manager.

        Note: This method is mainly for compatibility. In a multi-manager setup,
        each manager has its own checkpoint paths.
        """
        return self.managers[0].get_checkpoint_path(filename)

    def load_checkpoint(self, filename: str, model_class: type[T], **kwargs) -> Optional[List[T]]:
        """Load checkpoint using the configured strategy.

        Args:
            filename: Name of the checkpoint file
            model_class: Pydantic model class for deserializing the data
            **kwargs: Additional arguments passed to underlying managers

        Returns:
            List of model instances if checkpoint exists, None otherwise
        """
        if not self.enabled:
            return None

        if self.load_strategy == "first_found":
            # Try each manager until one succeeds
            for mgr in self.managers:
                if mgr.enabled:
                    result = mgr.load_checkpoint(filename, model_class, **kwargs)
                    if result is not None:
                        logger.info(
                            f"Loaded checkpoint '{filename}' from {type(mgr).__name__}"
                        )
                        return result

            logger.debug(f"No checkpoint '{filename}' found in any manager")
            return None

        elif self.load_strategy == "priority":
            # Only try the first enabled manager
            for mgr in self.managers:
                if mgr.enabled:
                    result = mgr.load_checkpoint(filename, model_class, **kwargs)
                    if result is not None:
                        logger.info(
                            f"Loaded checkpoint '{filename}' from primary manager {type(mgr).__name__}"
                        )
                    else:
                        logger.debug(
                            f"Checkpoint '{filename}' not found in primary manager"
                        )
                    return result

            return None

        else:
            raise ValueError(f"Unknown load strategy: {self.load_strategy}")

    def save_checkpoint(self, filename: str, data: List[T], **kwargs) -> None:
        """Save checkpoint using the configured strategy.

        Args:
            filename: Name of the checkpoint file
            data: List of model instances to save
            **kwargs: Additional arguments passed to underlying managers
        """
        if not self.enabled:
            return

        if self.save_strategy == "all_enabled":
            # Save to all enabled managers
            saved_to = []
            for mgr in self.managers:
                if mgr.enabled:
                    mgr.save_checkpoint(filename, data, **kwargs)
                    saved_to.append(type(mgr).__name__)

            if saved_to:
                logger.info(
                    f"Saved checkpoint '{filename}' to {len(saved_to)} managers: {', '.join(saved_to)}"
                )

        elif self.save_strategy == "primary_only":
            # Save only to the first enabled manager
            for mgr in self.managers:
                if mgr.enabled:
                    mgr.save_checkpoint(filename, data, **kwargs)
                    logger.info(
                        f"Saved checkpoint '{filename}' to primary manager {type(mgr).__name__}"
                    )
                    break

        else:
            raise ValueError(f"Unknown save strategy: {self.save_strategy}")

    def list_checkpoints(self) -> List[str]:
        """List all available checkpoint files across all managers.

        Returns:
            Sorted list of unique checkpoint filenames
        """
        if not self.enabled:
            return []

        all_checkpoints = set()
        for mgr in self.managers:
            if mgr.enabled and hasattr(mgr, "list_checkpoints"):
                checkpoints = mgr.list_checkpoints()
                all_checkpoints.update(checkpoints)

        return sorted(list(all_checkpoints))

    def delete_checkpoint(self, filename: str) -> bool:
        """Delete a checkpoint file from all managers.

        Args:
            filename: Name of the checkpoint file to delete

        Returns:
            True if deleted from at least one manager, False otherwise
        """
        if not self.enabled:
            return False

        deleted_from = []
        for mgr in self.managers:
            if mgr.enabled and hasattr(mgr, "delete_checkpoint"):
                if mgr.delete_checkpoint(filename):
                    deleted_from.append(type(mgr).__name__)

        if deleted_from:
            logger.info(
                f"Deleted checkpoint '{filename}' from {len(deleted_from)} managers: {', '.join(deleted_from)}"
            )
            return True

        return False

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the multi-checkpoint setup.

        Returns:
            Dictionary with manager statistics
        """
        stats = {
            "enabled": self.enabled,
            "save_strategy": self.save_strategy,
            "load_strategy": self.load_strategy,
            "num_managers": len(self.managers),
            "enabled_managers": sum(1 for mgr in self.managers if mgr.enabled),
            "managers": [],
        }

        for mgr in self.managers:
            mgr_info = {
                "type": type(mgr).__name__,
                "enabled": mgr.enabled,
                "checkpoint_dir": mgr.checkpoint_dir,
            }

            # Add checkpoint count if manager supports listing
            if hasattr(mgr, "list_checkpoints"):
                mgr_info["checkpoint_count"] = len(mgr.list_checkpoints())

            stats["managers"].append(mgr_info)

        return stats
