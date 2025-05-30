# Procedural API Tutorial

This tutorial demonstrates how to use Kura's procedural API for step-by-step conversation analysis with full control over each processing stage.

## Overview

The procedural API provides a functional approach to the Kura pipeline, allowing you to:
- Process conversations step by step
- Use different models for each stage
- Leverage checkpoint management
- Visualize results in multiple formats

## Prerequisites

Before starting, ensure you have:
- Kura installed with development dependencies
- Access to an OpenAI API key (set as `OPENAI_API_KEY` environment variable)

## Step 1: Import Required Modules

```python
from kura.v1 import (
    summarise_conversations,
    generate_base_clusters_from_conversation_summaries,
    reduce_clusters_from_base_clusters,
    reduce_dimensionality_from_clusters,
    CheckpointManager,
)

from kura.v1.visualization import (
    visualise_clusters_enhanced,
    visualise_clusters_rich,
    visualise_from_checkpoint_manager,
    visualise_pipeline_results,
)

from kura.types import Conversation
from kura.summarisation import SummaryModel
from kura.cluster import ClusterModel
from kura.meta_cluster import MetaClusterModel
from kura.dimensionality import HDBUMAP
```

## Step 2: Initialize Models

Set up individual models for each stage of the pipeline:

```python
from rich.console import Console

console = Console()
summary_model = SummaryModel(console=console)
cluster_model = ClusterModel(console=console)
meta_cluster_model = MetaClusterModel(console=console)
dimensionality_model = HDBUMAP()
```

## Step 3: Set Up Checkpointing

Enable checkpoint management to save intermediate results:

```python
checkpoint_manager = CheckpointManager("./tutorial_checkpoints", enabled=True)
```

## Step 4: Load Conversations

Load conversations from a Hugging Face dataset:

```python
conversations = Conversation.from_hf_dataset(
    "ivanleomk/synthetic-gemini-conversations",
    split="train"
)
```

## Step 5: Process Conversations

Process the conversations through each stage of the pipeline:

```python
import asyncio

async def process_conversations():
    # Step 1: Generate summaries
    summaries = await summarise_conversations(
        conversations,
        model=summary_model,
        checkpoint_manager=checkpoint_manager
    )

    # Step 2: Create base clusters
    clusters = await generate_base_clusters_from_conversation_summaries(
        summaries,
        model=cluster_model,
        checkpoint_manager=checkpoint_manager
    )

    # Step 3: Reduce to meta clusters
    reduced_clusters = await reduce_clusters_from_base_clusters(
        clusters,
        model=meta_cluster_model,
        checkpoint_manager=checkpoint_manager
    )

    # Step 4: Project to 2D
    projected_clusters = await reduce_dimensionality_from_clusters(
        reduced_clusters,
        model=dimensionality_model,
        checkpoint_manager=checkpoint_manager,
    )

    return reduced_clusters, projected_clusters

# Run the pipeline
reduced_clusters, projected_clusters = asyncio.run(process_conversations())
```

## Step 6: Visualize Results

The procedural API offers multiple visualization options:

### Basic Visualization
```python
visualise_from_checkpoint_manager(
    checkpoint_manager,
    meta_cluster_model,
    style="basic"
)
```

### Enhanced Visualization
```python
visualise_pipeline_results(
    reduced_clusters,
    style="enhanced"
)
```

### Rich Console Visualization
```python
visualise_clusters_rich(
    reduced_clusters,
    console=console
)
```

### Direct Checkpoint Visualization
```python
checkpoint_path = checkpoint_manager.get_checkpoint_path(
    meta_cluster_model.checkpoint_filename
)
visualise_clusters_enhanced(checkpoint_path=checkpoint_path)
```

## Complete Example

Here's a complete working example that processes conversations and displays results:

```python
import asyncio
from rich.console import Console
from kura.v1 import (
    summarise_conversations,
    generate_base_clusters_from_conversation_summaries,
    reduce_clusters_from_base_clusters,
    reduce_dimensionality_from_clusters,
    CheckpointManager,
)
from kura.v1.visualization import visualise_pipeline_results
from kura.types import Conversation
from kura.summarisation import SummaryModel
from kura.cluster import ClusterModel
from kura.meta_cluster import MetaClusterModel
from kura.dimensionality import HDBUMAP

async def main():
    # Initialize models
    console = Console()
    summary_model = SummaryModel(console=console)
    cluster_model = ClusterModel(console=console)
    meta_cluster_model = MetaClusterModel(console=console)
    dimensionality_model = HDBUMAP()

    # Set up checkpointing
    checkpoint_manager = CheckpointManager("./checkpoints", enabled=True)

    # Load conversations
    conversations = Conversation.from_hf_dataset(
        "ivanleomk/synthetic-gemini-conversations",
        split="train"
    )

    # Process through pipeline
    summaries = await summarise_conversations(
        conversations,
        model=summary_model,
        checkpoint_manager=checkpoint_manager
    )

    clusters = await generate_base_clusters_from_conversation_summaries(
        summaries,
        model=cluster_model,
        checkpoint_manager=checkpoint_manager
    )

    reduced_clusters = await reduce_clusters_from_base_clusters(
        clusters,
        model=meta_cluster_model,
        checkpoint_manager=checkpoint_manager
    )

    projected_clusters = await reduce_dimensionality_from_clusters(
        reduced_clusters,
        model=dimensionality_model,
        checkpoint_manager=checkpoint_manager,
    )

    # Visualize results
    visualise_pipeline_results(reduced_clusters, style="enhanced")

    print(f"\\nProcessed {len(conversations)} conversations")
    print(f"Created {len(reduced_clusters)} meta clusters")
    print(f"Checkpoints saved to: {checkpoint_manager.checkpoint_dir}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Benefits of the Procedural API

1. **Fine-grained Control**: Process each step independently
2. **Flexibility**: Mix and match different model implementations
3. **Checkpoint Management**: Resume from any stage
4. **Multiple Visualization Options**: Choose the best format for your needs
5. **Functional Programming**: No hidden state, clear data flow

## Next Steps

- Explore the [API Reference](../api/index.md) for detailed documentation
- Learn about [core concepts](../core-concepts/overview.md)
