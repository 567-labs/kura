from abc import ABC, abstractmethod
from typing import List, Optional, Union, TYPE_CHECKING, Any
from pathlib import Path
from kura.types import Cluster

if TYPE_CHECKING:
    from rich.console import Console as ConsoleType
else:
    ConsoleType = Any


class BaseVisualizationModel(ABC):
    """Base class for cluster visualization models.
    
    This abstract base class defines the interface that all visualization models
    must implement. It follows the procedural API design where models accept
    clusters directly rather than being tightly coupled to Kura instances.
    """
    
    @property
    @abstractmethod
    def checkpoint_filename(self) -> str:
        """The filename to use for checkpointing clusters for this visualization model."""
        pass

    @abstractmethod
    def visualize_clusters(
        self,
        clusters: Optional[List[Cluster]] = None,
        *,
        checkpoint_path: Optional[Union[str, Path]] = None,
        style: str = "basic",
        console: Optional["ConsoleType"] = None,
    ) -> None:
        """Visualize a list of clusters.
        
        Args:
            clusters: List of clusters to visualize. If None, loads from checkpoint_path
            checkpoint_path: Path to checkpoint file to load clusters from
            style: Visualization style (implementation-specific)
            console: Console instance for rich output (if supported)
            
        Raises:
            ValueError: If neither clusters nor checkpoint_path is provided
            FileNotFoundError: If checkpoint file doesn't exist
        """
        pass
