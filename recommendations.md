# Logging Strategy: Supporting Weights & Biases and Other Providers

## Current State

The codebase uses Python's standard `logging` module throughout. This provides basic logging but doesn't support experiment tracking that tools like Weights & Biases offer.

## Recommended Approach

Following Kura's procedural design, implement logging as **configurable, injectable dependencies** with a unified interface.

### 1. Unified Logger Interface

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union

class BaseLogger(ABC):
    """Unified interface for both application logging and experiment tracking."""
    
    # Standard logging methods
    @abstractmethod
    def info(self, message: str, **kwargs) -> None:
        pass
    
    @abstractmethod
    def debug(self, message: str, **kwargs) -> None:
        pass
    
    @abstractmethod
    def warning(self, message: str, **kwargs) -> None:
        pass
    
    @abstractmethod
    def error(self, message: str, **kwargs) -> None:
        pass
    
    # Experiment tracking methods (optional - can be no-ops)
    def log_metrics(self, metrics: Dict[str, Union[int, float]], step: Optional[int] = None) -> None:
        """Log experiment metrics. Default: no-op."""
        pass
    
    def log_parameters(self, params: Dict[str, Any]) -> None:
        """Log experiment parameters. Default: no-op."""
        pass
    
    def start_run(self, run_name: Optional[str] = None, **kwargs) -> None:
        """Start experiment run. Default: no-op."""
        pass
    
    def finish_run(self) -> None:
        """Finish experiment run. Default: no-op."""
        pass
```

### 2. Standard Logger Implementation

```python
class StandardLogger(BaseLogger):
    """Standard Python logging - ignores experiment methods."""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
    
    def info(self, message: str, **kwargs) -> None:
        self.logger.info(message, extra=kwargs)
    
    def debug(self, message: str, **kwargs) -> None:
        self.logger.debug(message, extra=kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        self.logger.warning(message, extra=kwargs)
    
    def error(self, message: str, **kwargs) -> None:
        self.logger.error(message, extra=kwargs)
    
    # Experiment methods are no-ops (inherited defaults)
```

### 3. Multiple Provider Implementations

The interface is flexible enough for different experiment tracking providers:

```python
class WandbLogger(BaseLogger):
    """Weights & Biases logger."""
    
    def __init__(self, name: str, project: str, entity: Optional[str] = None):
        self.logger = logging.getLogger(name)
        self.project = project
        self.entity = entity
    
    def info(self, message: str, **kwargs) -> None:
        self.logger.info(message, extra=kwargs)
    # ... other logging methods
    
    def start_run(self, run_name: Optional[str] = None, **kwargs) -> None:
        import wandb
        wandb.init(project=self.project, entity=self.entity, name=run_name, **kwargs)
    
    def log_metrics(self, metrics: Dict[str, Union[int, float]], step: Optional[int] = None) -> None:
        import wandb
        wandb.log(metrics, step=step)
    
    def log_parameters(self, params: Dict[str, Any]) -> None:
        import wandb
        wandb.config.update(params)
    
    def finish_run(self) -> None:
        import wandb
        wandb.finish()

class MLflowLogger(BaseLogger):
    """MLflow logger."""
    
    def __init__(self, name: str, experiment_name: str, tracking_uri: Optional[str] = None):
        self.logger = logging.getLogger(name)
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
    
    def info(self, message: str, **kwargs) -> None:
        self.logger.info(message, extra=kwargs)
    # ... other logging methods
    
    def start_run(self, run_name: Optional[str] = None, **kwargs) -> None:
        import mlflow
        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        mlflow.start_run(run_name=run_name, **kwargs)
    
    def log_metrics(self, metrics: Dict[str, Union[int, float]], step: Optional[int] = None) -> None:
        import mlflow
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)
    
    def log_parameters(self, params: Dict[str, Any]) -> None:
        import mlflow
        mlflow.log_params(params)
    
    def finish_run(self) -> None:
        import mlflow
        mlflow.end_run()

class NeptuneLogger(BaseLogger):
    """Neptune logger."""
    
    def __init__(self, name: str, project: str, api_token: Optional[str] = None):
        self.logger = logging.getLogger(name)
        self.project = project
        self.api_token = api_token
        self.run = None
    
    def info(self, message: str, **kwargs) -> None:
        self.logger.info(message, extra=kwargs)
    # ... other logging methods
    
    def start_run(self, run_name: Optional[str] = None, **kwargs) -> None:
        import neptune
        self.run = neptune.init_run(
            project=self.project,
            api_token=self.api_token,
            name=run_name,
            **kwargs
        )
    
    def log_metrics(self, metrics: Dict[str, Union[int, float]], step: Optional[int] = None) -> None:
        if self.run:
            for key, value in metrics.items():
                self.run[key].log(value, step=step)
    
    def log_parameters(self, params: Dict[str, Any]) -> None:
        if self.run:
            self.run["parameters"] = params
    
    def finish_run(self) -> None:
        if self.run:
            self.run.stop()
```

### 4. Update Function Signatures

Replace hardcoded loggers with injectable unified logger:

```python
async def generate_base_clusters_from_conversation_summaries(
    summaries: List[ConversationSummary],
    embedding_model: BaseEmbeddingModel = OpenAIEmbeddingModel(),
    clustering_method: BaseClusteringMethod = KmeansClusteringModel(),
    clustering_model: BaseClusterDescriptionModel = ClusterDescriptionModel(),
    # NEW: Unified logger parameter
    logger: BaseLogger = StandardLogger(__name__),
    run_name: Optional[str] = None,
    checkpoint_manager: Optional[CheckpointManager] = None,
    **kwargs,
) -> List[Cluster]:
    """Cluster conversation summaries with configurable logging."""
    
    # Start experiment run (no-op for StandardLogger)
    logger.start_run(run_name=run_name)
    
    # Log parameters (no-op for StandardLogger)
    logger.log_parameters({
        "num_summaries": len(summaries),
        "embedding_model": type(embedding_model).__name__,
        "clustering_method": type(clustering_method).__name__,
        **kwargs
    })
    
    logger.info(f"Starting cluster generation with {len(summaries)} summaries")
    
    try:
        # ... existing clustering logic ...
        
        # Log results (no-op for StandardLogger)
        logger.log_metrics({
            "num_clusters": len(clusters),
            "avg_cluster_size": sum(len(c.chat_ids) for c in clusters) / len(clusters),
        })
        
        logger.info(f"Generated {len(clusters)} clusters")
        return clusters
        
    except Exception as e:
        logger.error(f"Failed cluster generation: {e}")
        raise
    finally:
        logger.finish_run()
```

## Usage Examples

### Default Usage (Standard Logging)
```python
# Uses StandardLogger by default - same as current behavior
clusters = await generate_base_clusters_from_conversation_summaries(summaries)
```

### With Different Experiment Trackers

```python
# Weights & Biases
wandb_logger = WandbLogger(__name__, project="kura-clustering", entity="my-team")
clusters = await generate_base_clusters_from_conversation_summaries(
    summaries, logger=wandb_logger, run_name="clustering-v1"
)

# MLflow
mlflow_logger = MLflowLogger(__name__, experiment_name="kura-clustering")
clusters = await generate_base_clusters_from_conversation_summaries(
    summaries, logger=mlflow_logger, run_name="clustering-v1"
)

# Neptune
neptune_logger = NeptuneLogger(__name__, project="team/kura-clustering")
clusters = await generate_base_clusters_from_conversation_summaries(
    summaries, logger=neptune_logger, run_name="clustering-v1"
)
```

## Implementation Plan

1. **Add interface**: Create `BaseLogger` with unified methods
2. **Implement Standard**: Create `StandardLogger` (backwards compatible)
3. **Implement WandB**: Create `WandbLogger` class  
4. **Update functions**: Replace `logger = logging.getLogger(__name__)` with injectable `logger: BaseLogger`
5. **Add to dependencies**: Make wandb an optional dependency

## Benefits

- **Unified Interface**: One logger handles both regular logging AND experiment tracking
- **Backwards Compatible**: `StandardLogger` works exactly like current logging
- **Drop-in Replacement**: Just change the logger parameter to switch providers
- **No Conditionals**: No `if experiment_logger:` checks needed
- **Testable**: Simple to mock the entire logger interface
