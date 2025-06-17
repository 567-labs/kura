import logging
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from kura.base_classes import BaseClusterLogger


class StandardLogger(BaseClusterLogger):
    """
    Default logger implementation using Python's standard logging.
    
    This logger provides a simple, dependency-free way to log clustering 
    experiments using Python's built-in logging module. It structures data
    as JSON where possible and provides basic artifact support by copying
    files to a specified directory.
    
    Example:
        >>> logger = StandardLogger("clustering_experiment")
        >>> with logger:
        ...     logger.log_params({"n_clusters": 8, "algorithm": "kmeans"})
        ...     logger.log_metrics({"silhouette_score": 0.42})
        ...     logger.log_artifact("plot.png", "cluster_visualization")
    """
    
    def __init__(
        self, 
        name: str = "kura_clustering",
        level: int = logging.INFO,
        log_dir: Optional[Union[str, Path]] = None,
        artifact_dir: Optional[Union[str, Path]] = None
    ):
        """
        Initialize the standard logger.
        
        Args:
            name: Name for the logger instance
            level: Logging level (default: INFO)
            log_dir: Directory for log files (default: None, uses console only)
            artifact_dir: Directory to copy artifacts to (default: None, disables artifacts)
        """
        self.name = name
        self.logger = logging.getLogger(f"kura.clustering.{name}")
        self.logger.setLevel(level)
        
        # Avoid duplicate handlers if logger already configured
        if not self.logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
            
            # File handler if log_dir specified
            if log_dir:
                log_dir = Path(log_dir)
                log_dir.mkdir(parents=True, exist_ok=True)
                
                file_handler = logging.FileHandler(
                    log_dir / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
                )
                file_handler.setLevel(level)
                file_formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
                file_handler.setFormatter(file_formatter)
                self.logger.addHandler(file_handler)
        
        # Artifact handling
        self.artifact_dir = Path(artifact_dir) if artifact_dir else None
        if self.artifact_dir:
            self.artifact_dir.mkdir(parents=True, exist_ok=True)
            
        self.run_active = False
        self.run_metadata = {}

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log clustering parameters and configuration."""
        try:
            params_json = json.dumps(params, indent=2, default=str)
            self.logger.info(f"PARAMS: {params_json}")
        except (TypeError, ValueError) as e:
            # Fallback for non-serializable objects
            self.logger.info(f"PARAMS: {params} (JSON serialization failed: {e})")

    def log_metrics(self, metrics: Dict[str, Union[float, int]], step: Optional[int] = None) -> None:
        """Log clustering metrics and performance stats."""
        try:
            metrics_data = {"metrics": metrics}
            if step is not None:
                metrics_data["step"] = step
            metrics_json = json.dumps(metrics_data, indent=2)
            self.logger.info(f"METRICS: {metrics_json}")
        except (TypeError, ValueError) as e:
            # Fallback for non-serializable objects
            step_info = f" (step {step})" if step is not None else ""
            self.logger.info(f"METRICS{step_info}: {metrics} (JSON serialization failed: {e})")

    def log_errors(self, error: Union[str, Exception], context: Optional[Dict[str, Any]] = None) -> None:
        """Log errors with optional context."""
        error_msg = str(error)
        
        if isinstance(error, Exception):
            # Log with exception info for proper stack traces
            if context:
                try:
                    context_json = json.dumps(context, indent=2, default=str)
                    self.logger.error(f"ERROR with context: {context_json}", exc_info=error)
                except (TypeError, ValueError):
                    self.logger.error(f"ERROR with context: {context}", exc_info=error)
            else:
                self.logger.error(f"ERROR: {error_msg}", exc_info=error)
        else:
            # String error message
            if context:
                try:
                    context_json = json.dumps(context, indent=2, default=str)
                    self.logger.error(f"ERROR: {error_msg} | Context: {context_json}")
                except (TypeError, ValueError):
                    self.logger.error(f"ERROR: {error_msg} | Context: {context}")
            else:
                self.logger.error(f"ERROR: {error_msg}")

    def log(self, data: Any, key: str, **metadata) -> None:
        """Generic logging method for arbitrary data."""
        try:
            # Try to structure the data nicely
            log_data = {"key": key, "data": data}
            if metadata:
                log_data["metadata"] = metadata
            
            log_json = json.dumps(log_data, indent=2, default=str)
            self.logger.info(f"LOG: {log_json}")
        except (TypeError, ValueError) as e:
            # Fallback for non-serializable objects
            metadata_str = f" | Metadata: {metadata}" if metadata else ""
            self.logger.info(f"LOG [{key}]: {data}{metadata_str} (JSON serialization failed: {e})")

    def log_artifact(self, file_path: Union[str, Path], name: Optional[str] = None, **metadata) -> None:
        """Log files by copying them to artifact directory if available."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            self.log_errors(f"Artifact file does not exist: {file_path}")
            return
            
        artifact_name = name or file_path.name
        
        if self.artifact_dir:
            try:
                # Copy file to artifact directory
                dest_path = self.artifact_dir / artifact_name
                shutil.copy2(file_path, dest_path)
                
                # Log the successful artifact copy
                artifact_info = {
                    "name": artifact_name,
                    "source_path": str(file_path),
                    "artifact_path": str(dest_path),
                    "size_bytes": file_path.stat().st_size
                }
                artifact_info.update(metadata)
                
                self.log(artifact_info, "artifact_copied")
                
            except Exception as e:
                self.log_errors(f"Failed to copy artifact {file_path}: {e}", 
                              {"artifact_name": artifact_name, **metadata})
        else:
            # Fall back to logging file path
            super().log_artifact(file_path, name, **metadata)

    def supports_artifacts(self) -> bool:
        """Check if artifact directory is configured."""
        return self.artifact_dir is not None

    def start_run(self, name: str, **metadata) -> None:
        """Start a new logging run."""
        self.run_active = True
        self.run_metadata = metadata
        
        run_info = {
            "run_name": name,
            "start_time": datetime.now().isoformat(),
            "metadata": metadata
        }
        
        self.logger.info("=" * 60)
        self.logger.info(f"STARTING RUN: {name}")
        self.logger.info("=" * 60)
        
        self.log(run_info, "run_started")

    def end_run(self) -> None:
        """End the current logging run."""
        if self.run_active:
            end_info = {
                "end_time": datetime.now().isoformat(),
                "metadata": self.run_metadata
            }
            
            self.log(end_info, "run_ended")
            
            self.logger.info("=" * 60)
            self.logger.info("RUN COMPLETED")
            self.logger.info("=" * 60)
            
            self.run_active = False
            self.run_metadata = {}

    @property
    def checkpoint_filename(self) -> str:
        """Return filename for checkpointing logger configuration."""
        return f"logger_config_{self.name}.json"


def create_logger(
    provider: str = "standard",
    name: str = "kura_clustering",
    **kwargs
) -> BaseClusterLogger:
    """
    Factory function to create logger instances.
    
    Args:
        provider: Logger provider type (currently only "standard" supported)
        name: Name for the logger instance
        **kwargs: Provider-specific configuration
        
    Returns:
        Configured logger instance
        
    Example:
        >>> # Basic usage
        >>> logger = create_logger("standard", "my_experiment")
        
        >>> # With file logging and artifacts
        >>> logger = create_logger(
        ...     "standard", 
        ...     "my_experiment",
        ...     log_dir="./logs",
        ...     artifact_dir="./artifacts"
        ... )
    """
    if provider == "standard":
        return StandardLogger(name=name, **kwargs)
    else:
        raise ValueError(f"Unsupported logger provider: {provider}. Currently supported: ['standard']")