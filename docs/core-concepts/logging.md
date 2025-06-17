# Logging

Kura provides a unified logging interface for tracking clustering experiments, metrics, and artifacts. The logging system is designed to be extensible, allowing different providers while maintaining a consistent API.

## Overview

The logging system consists of:

- **BaseClusterLogger**: Abstract base class defining the interface
- **StandardLogger**: Default implementation using Python's standard logging
- **create_logger()**: Factory function for creating logger instances

## Quick Start

```python
from kura import create_logger

# Basic usage with console logging
logger = create_logger("standard", "my_experiment")

with logger:
    # Log experiment parameters
    logger.log_params({
        "n_clusters": 8,
        "algorithm": "kmeans",
        "embedding_model": "text-embedding-3-large"
    })
    
    # Log metrics during processing
    logger.log_metrics({
        "silhouette_score": 0.42,
        "inertia": 2847.3,
        "processing_time": 45.2
    })
    
    # Log general information
    logger.log("Processing 1500 conversations", "status")
    
    # Log artifacts (if supported)
    if logger.supports_artifacts():
        logger.log_artifact("cluster_plot.png", "visualization")
    
    # Log errors with context
    try:
        risky_operation()
    except Exception as e:
        logger.log_errors(e, {
            "operation": "clustering",
            "data_size": 1500
        })
```

## Core Interface

All logger providers must implement these four required methods:

### log_params(params)

Log clustering parameters and configuration:

```python
logger.log_params({
    "n_clusters": 8,
    "algorithm": "kmeans",
    "embedding_model": "text-embedding-3-large",
    "random_state": 42
})
```

### log_metrics(metrics, step=None)

Log numerical metrics with optional step tracking:

```python
# Basic metrics
logger.log_metrics({
    "silhouette_score": 0.42,
    "inertia": 2847.3
})

# Time series metrics
for i, score in enumerate(scores):
    logger.log_metrics({"validation_score": score}, step=i)
```

### log_errors(error, context=None)

Log errors with optional context for debugging:

```python
try:
    cluster_data()
except Exception as e:
    logger.log_errors(e, {
        "operation": "clustering",
        "data_size": len(conversations),
        "algorithm": "kmeans"
    })
```

### log(data, key, **metadata)

Generic method for logging arbitrary data:

```python
logger.log("Starting embedding phase", "status")
logger.log(cluster_assignments, "results", algorithm="kmeans")
logger.log({"progress": 0.75}, "progress", timestamp=datetime.now())
```

## Optional Features

### Artifact Support

Some providers support file artifacts:

```python
# Check if artifacts are supported
if logger.supports_artifacts():
    logger.log_artifact("cluster_plot.png", "visualization")
    logger.log_artifact("embeddings.npy", "embeddings")
else:
    # Fallback to logging file paths
    logger.log("Cluster plot saved to cluster_plot.png", "artifact_info")
```

### Clustering-Specific Helpers

Convenience methods for common clustering tasks:

```python
# Log cluster summaries
cluster_data = {
    0: {"size": 45, "coherence": 0.8, "top_terms": ["error", "bug"]},
    1: {"size": 32, "coherence": 0.7, "top_terms": ["feature", "request"]}
}
logger.log_cluster_summary(cluster_data)

# Log sample conversations from each cluster
sample_conversations = ["How do I reset my password?", "Login not working"]
logger.log_conversation_sample(cluster_id=0, conversations=sample_conversations)
```

## Standard Logger

The default `StandardLogger` uses Python's built-in logging module:

```python
from kura import StandardLogger

# Console logging only
logger = StandardLogger("experiment_name")

# With file logging
logger = StandardLogger(
    "experiment_name",
    log_dir="./logs"
)

# With artifact support
logger = StandardLogger(
    "experiment_name",
    log_dir="./logs",
    artifact_dir="./artifacts"
)
```

### Features

- **Console output**: Formatted log messages to stdout/stderr
- **File logging**: Optional log files with timestamps
- **JSON structuring**: Attempts to serialize data as JSON when possible
- **Artifact copying**: Copies files to artifact directory if configured
- **Error handling**: Proper exception logging with stack traces

### Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | "kura_clustering" | Logger instance name |
| `level` | int | logging.INFO | Logging level |
| `log_dir` | str/Path | None | Directory for log files |
| `artifact_dir` | str/Path | None | Directory for artifacts |

## Context Management

Loggers support context managers for run lifecycle:

```python
with logger:
    logger.start_run("clustering_experiment_v1", 
                    dataset="customer_feedback",
                    version="1.0")
    
    # Your clustering code here...
    
    # Automatically calls end_run() and logs any exceptions
```

Manual run management:

```python
logger.start_run("experiment_name", dataset="test_data")
try:
    # Your code here
    pass
finally:
    logger.end_run()
```

## Integration with Kura Pipeline

The logger can be integrated into existing Kura workflows:

```python
from kura import create_logger, Kura

# Set up logging
logger = create_logger("standard", "customer_analysis",
                      log_dir="./logs", artifact_dir="./artifacts")

with logger:
    # Log experiment configuration
    logger.log_params({
        "embedding_model": "text-embedding-3-large",
        "n_clusters": 10,
        "algorithm": "kmeans"
    })
    
    # Run Kura pipeline
    kura = Kura()
    results = kura.run(conversations)
    
    # Log results
    logger.log_metrics({
        "n_clusters_found": len(results.clusters),
        "silhouette_score": results.metrics.silhouette_score
    })
    
    # Save and log artifacts
    results.save_plot("cluster_visualization.png")
    logger.log_artifact("cluster_visualization.png", "clusters")
```

## Extending with Custom Providers

To add support for other logging providers (W&B, MLflow, etc.), implement the `BaseClusterLogger` interface:

```python
from kura.base_classes import BaseClusterLogger

class CustomLogger(BaseClusterLogger):
    def log_params(self, params):
        # Your implementation
        pass
    
    def log_metrics(self, metrics, step=None):
        # Your implementation  
        pass
    
    def log_errors(self, error, context=None):
        # Your implementation
        pass
    
    def log(self, data, key, **metadata):
        # Your implementation
        pass
    
    @property
    def checkpoint_filename(self):
        return "custom_logger_config.json"
```

## Best Practices

1. **Use context managers**: Always use `with` statements or explicit `start_run()`/`end_run()` calls
2. **Structure your data**: Use dictionaries for parameters and metrics when possible
3. **Provide context for errors**: Include relevant information when logging errors
4. **Check artifact support**: Use `supports_artifacts()` before calling `log_artifact()`
5. **Be consistent with naming**: Use clear, consistent keys for your logged data

## Troubleshooting

### JSON Serialization Errors

The StandardLogger attempts JSON serialization but falls back gracefully:

```python
# This will work even if some_object isn't JSON serializable
logger.log_params({"config": some_object})
```

### File Permission Issues

Ensure your process has write permissions to log and artifact directories:

```python
import os
os.makedirs("./logs", exist_ok=True)
os.makedirs("./artifacts", exist_ok=True)

logger = StandardLogger("test", log_dir="./logs", artifact_dir="./artifacts")
```

### Missing Artifacts

Check that files exist before logging them:

```python
from pathlib import Path

plot_path = Path("cluster_plot.png")
if plot_path.exists():
    logger.log_artifact(plot_path, "visualization")
else:
    logger.log_errors(f"Plot file not found: {plot_path}")
```