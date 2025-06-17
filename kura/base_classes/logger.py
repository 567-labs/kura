from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from pathlib import Path


class BaseClusterLogger(ABC):
    """
    Abstract base class for logging clustering experiments and analysis.
    
    This interface provides a unified way to log clustering parameters, metrics,
    errors, and artifacts across different logging providers (W&B, Braintrust, 
    Logfire, standard logging, etc.).
    
    Following the existing pattern in kura.base_classes, implementations should:
    - Accept provider-specific configuration in constructor
    - Provide clear error messages when features aren't supported
    - Use the context manager pattern for run lifecycle management
    """

    # Required Methods (All Providers Must Implement)
    
    @abstractmethod
    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Log clustering parameters and configuration.
        
        Args:
            params: Dictionary of configuration parameters (e.g., n_clusters,
                   algorithm, embedding_model, etc.)
        
        Example:
            >>> logger.log_params({
            ...     "n_clusters": 8,
            ...     "algorithm": "kmeans",
            ...     "embedding_model": "text-embedding-3-large"
            ... })
        """
        pass

    @abstractmethod  
    def log_metrics(self, metrics: Dict[str, Union[float, int]], step: Optional[int] = None) -> None:
        """
        Log clustering metrics and performance stats.
        
        Args:
            metrics: Dictionary of numerical metrics (e.g., silhouette_score, 
                    inertia, processing_time)
            step: Optional step number for time series tracking
        
        Example:
            >>> logger.log_metrics({
            ...     "silhouette_score": 0.42,
            ...     "inertia": 2847.3,
            ...     "processing_time": 45.2
            ... })
        """
        pass

    @abstractmethod
    def log_errors(self, error: Union[str, Exception], context: Optional[Dict[str, Any]] = None) -> None:
        """
        Log errors with optional context for debugging.
        
        Args:
            error: Error message string or Exception object
            context: Optional dictionary with additional context (operation, 
                    data_size, step, etc.)
        
        Example:
            >>> try:
            ...     risky_operation()
            ... except Exception as e:
            ...     logger.log_errors(e, {
            ...         "operation": "clustering",
            ...         "data_size": 1500,
            ...         "step": "embedding"
            ...     })
        """
        pass

    @abstractmethod
    def log(self, data: Any, key: str, **metadata) -> None:
        """
        Generic logging method for arbitrary data.
        
        Args:
            data: Any data to log (text, JSON, numbers, etc.)
            key: Identifier for the data
            **metadata: Additional metadata as keyword arguments
        
        Example:
            >>> logger.log("Processing 1500 conversations", "status")
            >>> logger.log(cluster_assignments, "cluster_results", 
            ...            algorithm="kmeans", timestamp=datetime.now())
        """
        pass

    # Optional Methods (Provider Capability-Dependent)
    
    def log_artifact(self, file_path: Union[str, Path], name: Optional[str] = None, **metadata) -> None:
        """
        Log files if provider supports artifacts.
        
        Default implementation logs file path if artifacts aren't supported.
        Override in concrete implementations that support file uploads.
        
        Args:
            file_path: Path to file to log
            name: Optional name for the artifact
            **metadata: Additional metadata for the artifact
        
        Example:
            >>> if logger.supports_artifacts():
            ...     logger.log_artifact("cluster_plot.png", "visualization")
            ... else:
            ...     logger.log("Cluster plot saved to cluster_plot.png", "artifact_info")
        """
        # Default fallback implementation
        file_path = Path(file_path)
        artifact_name = name or file_path.name
        self.log(f"Artifact saved: {file_path} ({artifact_name})", "artifact_info", **metadata)

    def supports_artifacts(self) -> bool:
        """
        Check if provider supports file artifacts.
        
        Returns:
            True if provider can upload/store files, False otherwise
        """
        return False

    # Clustering-Specific Helpers
    
    def log_cluster_summary(self, cluster_data: Dict[int, Dict[str, Any]]) -> None:
        """
        Log summary statistics for each cluster.
        
        Args:
            cluster_data: Dictionary mapping cluster IDs to their statistics
                         (e.g., size, representative conversations, quality metrics)
        
        Example:
            >>> cluster_summary = {
            ...     0: {"size": 45, "coherence": 0.8, "top_terms": ["error", "bug"]},
            ...     1: {"size": 32, "coherence": 0.7, "top_terms": ["feature", "request"]}
            ... }
            >>> logger.log_cluster_summary(cluster_summary)
        """
        for cluster_id, stats in cluster_data.items():
            self.log(stats, f"cluster_{cluster_id}_summary")

    def log_conversation_sample(self, cluster_id: int, conversations: List[str], max_samples: int = 3) -> None:
        """
        Log sample conversations from a cluster.
        
        Args:
            cluster_id: ID of the cluster
            conversations: List of conversation texts
            max_samples: Maximum number of conversations to log
        
        Example:
            >>> sample_convs = ["How do I reset password?", "Login not working", "Can't access account"]
            >>> logger.log_conversation_sample(0, sample_convs, max_samples=2)
        """
        samples = conversations[:max_samples]
        self.log(samples, f"cluster_{cluster_id}_samples", 
                total_conversations=len(conversations),
                samples_shown=len(samples))

    # Context Management
    
    def start_run(self, name: str, **metadata) -> None:
        """
        Start a new logging run/experiment.
        
        Args:
            name: Name for the run
            **metadata: Additional metadata for the run
        """
        self.log(f"Started run: {name}", "run_start", **metadata)

    def end_run(self) -> None:
        """End the current logging run/experiment."""
        self.log("Run completed", "run_end")

    def __enter__(self) -> 'BaseClusterLogger':
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        if exc_type is not None:
            self.log_errors(f"Run failed with {exc_type.__name__}: {exc_val}", 
                          {"exception_type": exc_type.__name__})
        self.end_run()

    @property
    @abstractmethod  
    def checkpoint_filename(self) -> str:
        """
        Return the filename to use for checkpointing this logger's configuration.
        
        Following the pattern from other base classes in kura.
        """
        pass