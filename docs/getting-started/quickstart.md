# Quickstart Guide

This guide will help you get started with Kura quickly using the procedural API for step-by-step conversation analysis.

## Prerequisites

Before you begin, make sure you have [installed Kura](installation.md) and set up your API key. Kura uses OpenAI by default, so you'll need to export your API key in your terminal.

## Complete Example

Here's a complete working example that you can copy and run immediately. This example processes conversations through Kura's four-stage pipeline: summarization, clustering, meta-clustering, and dimensionality reduction for visualization.

```python
import asyncio
from rich.console import Console
from kura import (
    ClusterDescriptionModel,
    SummaryModel,
    MetaClusterModel,
    Conversation,
    summarise_conversations,
    generate_base_clusters_from_conversation_summaries,
    reduce_clusters_from_base_clusters,
    reduce_dimensionality_from_clusters,
    visualise_pipeline_results,
)
from kura.checkpoints import JSONLCheckpointManager
from kura.types import Conversation
from kura.summarisation import SummaryModel
from kura.cluster import ClusterDescriptionModel
from kura.meta_cluster import MetaClusterModel
from kura.dimensionality import HDBUMAP


async def main():
    console = Console()

    # SummaryModel now supports caching to speed up re-runs!
    summary_model = SummaryModel(
        console=console,
        cache_dir="./.summary_cache",  # Optional: specify cache location
    )

    cluster_model = ClusterDescriptionModel(
        console=console,
    )
    meta_cluster_model = MetaClusterModel(console=console)
    dimensionality_model = HDBUMAP()

    # Set up checkpointing - you can choose from multiple backends
    # HuggingFace Datasets (advanced features, cloud sync)
    checkpoint_manager = JSONLCheckpointManager("./checkpoints/hf", enabled=True)

    # Load conversations from Hugging Face dataset
    conversations = Conversation.from_hf_dataset(
        "ivanleomk/synthetic-gemini-conversations", split="train"
    )
    summaries = await summarise_conversations(
        conversations, model=summary_model, checkpoint_manager=checkpoint_manager
    )
    clusters = await generate_base_clusters_from_conversation_summaries(
        summaries,
        model=cluster_model,
        checkpoint_manager=checkpoint_manager,
    )
    reduced_clusters = await reduce_clusters_from_base_clusters(
        clusters, checkpoint_manager=checkpoint_manager, model=meta_cluster_model
    )

    projected_clusters = await reduce_dimensionality_from_clusters(
        reduced_clusters,
        model=dimensionality_model,
        checkpoint_manager=checkpoint_manager,
    )

    # Visualize results
    visualise_pipeline_results(projected_clusters, style="basic")


if __name__ == "__main__":
    asyncio.run(main())
```

This example will process 190 synthetic programming conversations from Hugging Face, analyze them through the complete pipeline, and display the results in a clean tree structure showing how conversations are grouped into thematic clusters.

## Understanding the Pipeline

Now let's break down each section of the code above to understand what's happening at each stage.

### Model Initialization

!!! Note

    **💡 What's happening here**: We initialize four key models that each handle a specific transformation in the pipeline. The `console` parameter enables rich terminal output so you can see progress as the analysis runs.

```python
console = Console()
summary_model = SummaryModel(
    console=console,
    cache_dir="./.summary_cache",
)
cluster_model = ClusterDescriptionModel(console=console)
meta_cluster_model = MetaClusterModel(console=console)
dimensionality_model = HDBUMAP()
```

1. **[SummaryModel](../core-concepts/summarization.md)**: Converts raw conversations into structured summaries with key themes and intents
2. **[ClusterDescriptionModel](../core-concepts/clustering.md)**: Groups similar conversation summaries based on semantic content
3. **[MetaClusterModel](../core-concepts/meta-clustering.md)**: Creates hierarchical relationships between base clusters
4. **[HDBUMAP](../core-concepts/dimensionality-reduction.md)**: Reduces high-dimensional embeddings to 2D coordinates for visualization

### Checkpoint Management

!!! tip "Checkpoint Strategy"

    **What's happening here**: The checkpoint manager saves results after each pipeline stage, allowing you to resume processing or try different parameters without starting over. Choose JSONL for development, Parquet for production, or HuggingFace for advanced ML workflows.

Kura supports multiple checkpoint backends that allow you to save intermediate results and resume processing from any stage. This is essential for longer analyses where you might want to experiment with different clustering parameters without re-running the expensive summarization step, [read more about the different checkpoints we support](../core-concepts/checkpoints.md).

```python
checkpoint_manager = JSONLCheckpointManager("./checkpoints")
```

The example uses the [JSONLCheckpointManager](../core-concepts/checkpoints.md) which stores results in a human-readable JSONL format - perfect for development and debugging.

### Data Loading

!!! info "Data Loading"

    **What's happening here**: We load conversation data from a Hugging Face dataset. Each conversation contains the raw dialogue that will be processed through the pipeline.

Kura provides convenient methods for loading conversation data from various sources. The example uses a synthetic dataset from Hugging Face, but you can also load from local files, databases, or other sources.

```python
conversations = Conversation.from_hf_dataset(
    "ivanleomk/synthetic-gemini-conversations", split="train"
)
```

The [Conversation](../core-concepts/data-types.md) type provides a standardized format for dialogue data with built-in methods for loading from different sources and formats.

### Pipeline Execution

The core pipeline consists of four sequential transformations, each building on the results of the previous stage. This modular design allows you to mix and match different approaches at each stage.

> **💡 What's happening here**: Each function processes the output from the previous stage, progressively transforming raw conversations into organized, visualizable clusters.

```python
summaries = await summarise_conversations(
    conversations, model=summary_model, checkpoint_manager=checkpoint_manager
)
clusters = await generate_base_clusters_from_conversation_summaries(
    summaries, model=cluster_model, checkpoint_manager=checkpoint_manager,
)
reduced_clusters = await reduce_clusters_from_base_clusters(
    clusters, checkpoint_manager=checkpoint_manager, model=meta_cluster_model
)
projected_clusters = await reduce_dimensionality_from_clusters(
    reduced_clusters, model=dimensionality_model, checkpoint_manager=checkpoint_manager,
)
```

The [summarization stage](../core-concepts/summarization.md) extracts key themes and intents from each conversation. The [clustering stage](../core-concepts/clustering.md) groups similar summaries together. The [meta-clustering stage](../core-concepts/meta-clustering.md) creates hierarchical relationships between clusters. Finally, the [dimensionality reduction stage](../core-concepts/dimensionality-reduction.md) projects the high-dimensional cluster embeddings into 2D space for visualization.

### Visualization

The final step displays the results in a format that makes it easy to understand how conversations have been organized. Kura provides multiple visualization styles to suit different needs and contexts.

> **💡 What's happening here**: The visualization function takes the processed clusters and displays them as an organized tree structure showing the relationships between different conversation themes.

```python
visualise_pipeline_results(projected_clusters, style="basic")
```

The [visualization system](../core-concepts/visualization.md) supports three main styles: "basic" for clean output, "enhanced" for detailed statistics and progress bars, and "rich" for colorful interactive-style displays with comprehensive metrics.

## Customization Options

Kura's procedural design makes it easy to customize any part of the pipeline. You can swap out models, adjust parameters, or even skip certain stages depending on your specific needs.

For different clustering approaches, you can replace the `ClusterDescriptionModel` with alternatives like HDBSCAN for exploratory analysis or MiniBatch K-means for large datasets. The [clustering documentation](../core-concepts/clustering.md) provides detailed guidance on when to use each approach.

For different summary formats, you can configure the `SummaryModel` with custom prompts, response schemas, or even different language models. The [summarization documentation](../core-concepts/summarization.md) shows how to adapt the summarization process for specific domains or use cases.

## Using the Web Interface

For a more interactive experience, Kura includes a web interface that lets you explore results visually:

```bash
kura start-app --dir ./checkpoints
```

The web interface provides an interactive cluster map, hierarchical tree view, and detailed conversation dialog system for exploring your results in depth.

## Next Steps

Now that you've run your first analysis with Kura, you can explore the [core concepts](../core-concepts/overview.md) to understand how each component works, check out the [configuration guide](configuration.md) to customize Kura for your specific needs, or dive into the [API reference](../api/index.md) for detailed documentation of all available functions and options.
