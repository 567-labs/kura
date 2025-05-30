# Quickstart Guide

This guide will help you get started with Kura quickly using the procedural API for step-by-step conversation analysis.

## Overview

Kura provides a functional approach to conversation clustering that allows you to:

- Process conversations step by step with full control
- Use checkpoints to save intermediate results
- Visualize results in multiple formats

## Prerequisites

Before you begin, make sure you have:

1. [Installed Kura](installation.md)
2. Set up your API key (Kura uses OpenAI by default):
   ```bash
   export OPENAI_API_KEY=your_api_key_here
   ```

## Basic Workflow

Kura's workflow consists of four main steps:

1. **Summarization**: Generate concise summaries of conversations
2. **Base Clustering**: Group similar summaries together
3. **Meta Clustering**: Create hierarchical clusters for better organization
4. **Dimensionality Reduction**: Project clusters to 2D for visualization

## Complete Example

Here's a complete working example:

```python
import asyncio
from rich.console import Console
from kura import (
    summarise_conversations,
    generate_base_clusters_from_conversation_summaries,
    reduce_clusters_from_base_clusters,
    reduce_dimensionality_from_clusters,
    CheckpointManager,
)
from kura.visualization import visualise_pipeline_results
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

    # Set up checkpointing to save intermediate results
    checkpoint_manager = CheckpointManager("./checkpoints", enabled=True)

    # Load conversations from Hugging Face dataset
    conversations = Conversation.from_hf_dataset(
        "ivanleomk/synthetic-gemini-conversations",
        split="train"
    )

    # Process through the pipeline step by step
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

    print(f"\nProcessed {len(conversations)} conversations")
    print(f"Created {len(reduced_clusters)} meta clusters")
    print(f"Checkpoints saved to: {checkpoint_manager.checkpoint_dir}")

if __name__ == "__main__":
    asyncio.run(main())
```

This example will:

1. Load 190 synthetic programming conversations from Hugging Face
2. Process them through the complete analysis pipeline step by step
3. Generate hierarchical clusters organized into categories
4. Display the results with enhanced visualization

## Visualization Options & Output

Kura provides multiple visualization styles through the `visualise_pipeline_results` function. Simply change the `style` parameter to get different output formats:

```python
from kura.visualization import visualise_pipeline_results

# Choose from: "basic", "enhanced", or "rich"
visualise_pipeline_results(reduced_clusters, style="basic")
visualise_pipeline_results(reduced_clusters, style="enhanced")  # Recommended
visualise_pipeline_results(reduced_clusters, style="rich", console=console)
```

### Basic Style

Clean tree structure without extra formatting:

**Output:**

```
Clusters (190 conversations)
╠══ Generate SEO-optimized content for blogs and scripts (38 conversations)
║   ╠══ Assist in writing SEO-friendly blog posts (12 conversations)
║   ╚══ Help create SEO-driven marketing content (8 conversations)
╠══ Help analyze and visualize data with R and Tableau (25 conversations)
║   ╠══ Assist with data analysis and visualization in R (15 conversations)
║   ╚══ Troubleshoot sales data visualizations in Tableau (10 conversations)
... (and more clusters)
```

### Enhanced Style (Recommended)

Includes progress bars, statistics, and detailed formatting:

**Output:**

```
================================================================================
🎯 ENHANCED CLUSTER VISUALIZATION
================================================================================
🔸 📚 All Clusters (190 total conversations)
    📊 190 conversations (100.0%) [████████████████████]

╠══ 🔸 Generate SEO-optimized content for blogs and scripts
║   📊 38 conversations (20.0%) [████░░░░░░░░░░░░░░░░]
║   ╠══ 🔸 Assist in writing SEO-friendly blog posts
║   ║   📊 12 conversations (6.3%) [█░░░░░░░░░░░░░░░░░░░]
║   ╠══ 🔸 Write blog posts about diabetes medications
║   ║   📊 10 conversations (5.3%) [█░░░░░░░░░░░░░░░░░░░]
║   ╚══ 🔸 Help create SEO-driven marketing content
║       📊 8 conversations (4.2%) [░░░░░░░░░░░░░░░░░░░░]

╠══ 🔸 Help analyze and visualize data with R and Tableau
║   📊 25 conversations (13.2%) [██░░░░░░░░░░░░░░░░░░]
║   ╠══ 🔸 Assist with data analysis and visualization in R
║   ║   📊 15 conversations (7.9%) [█░░░░░░░░░░░░░░░░░░░]
║   ╚══ 🔸 Troubleshoot sales data visualizations in Tableau
║       📊 10 conversations (5.3%) [█░░░░░░░░░░░░░░░░░░░]

... (and more clusters)

================================================================================
📈 CLUSTER STATISTICS
================================================================================
📊 Total Clusters: 29
🌳 Root Clusters: 10
💬 Total Conversations: 190
📏 Average Conversations per Root Cluster: 19.0
================================================================================
```

### Rich Style

Colorful, interactive-style output with detailed descriptions and statistics tables:

**Output:**

```
╭──────────────────────────────────────────────────────────────────────────────╮
│                        🎯 RICH CLUSTER VISUALIZATION                         │
╰──────────────────────────────────────────────────────────────────────────────╯

📚 All Clusters (190 conversations)
├── Generate SEO-optimized content for blogs and scripts (38 conversations, 20.0%)
│   Users requested help in creating SEO-optimized blog posts and engaging
│   YouTube v...
│   Progress: [███░░░░░░░░░░░░]
│   ├── Assist in writing SEO-friendly blog posts (12 conversations, 6.3%)
│   │   The users sought assistance in crafting engaging and SEO-friendly blog
│   │   posts acr...
│   │   Progress: [░░░░░░░░░░░░░░░]
│   ├── Write blog posts about diabetes medications (10 conversations, 5.3%)
│   │   The users sought assistance in creating blog posts focused on diabetes
│   │   treatment...
│   │   Progress: [░░░░░░░░░░░░░░░]
│   └── Help create SEO-driven marketing content (8 conversations, 4.2%)
│       The users sought assistance in developing SEO-optimized marketing
│       content across...
│       Progress: [░░░░░░░░░░░░░░░]
├── Help analyze and visualize data with R and Tableau (25 conversations, 13.2%)
│   Users sought help with analyzing and visualizing datasets in both R and
│   Tableau,...
│   Progress: [█░░░░░░░░░░░░░░]
... (and more clusters)

       📈 Cluster Statistics                📊 Cluster Size Distribution
╭─────────────────────────┬───────╮  ╭────────────────────┬───────┬────────────╮
│ Metric                  │ Value │  │ Size Range         │ Count │ Percentage │
├─────────────────────────┼───────┤  ├────────────────────┼───────┼────────────┤
│ 📊 Total Clusters       │ 29    │  │ 🔥 Large (>100)    │ 0     │ 0.0%       │
│ 🌳 Root Clusters        │ 10    │  │ 📈 Medium (21-100) │ 3     │ 30.0%      │
│ 💬 Total Conversations  │ 190   │  │ 📊 Small (6-20)    │ 7     │ 70.0%      │
│ 📏 Avg per Root Cluster │ 19.0  │  │ 🔍 Tiny (1-5)      │ 0     │ 0.0%       │
╰─────────────────────────┴───────╯  ╰────────────────────┴───────┴────────────╯
```

## Using the Web Interface

For a more interactive experience, Kura includes a web interface:

```bash
# Start with default checkpoint directory
kura start-app

# Or use a custom checkpoint directory
kura start-app --dir ./checkpoints
```

Expected output:

```
🚀 Access website at (http://localhost:8000)

INFO:     Started server process [14465]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

Access the web interface at http://localhost:8000 to explore:

- **Cluster Map**: 2D visualization of conversation clusters
- **Cluster Tree**: Hierarchical view of cluster relationships
- **Cluster Details**: In-depth information about selected clusters
- **Conversation Dialog**: Examine individual conversations
- **Metadata Filtering**: Filter clusters based on extracted properties

## Benefits of the Procedural API

1. **Fine-grained Control**: Process each step independently
2. **Flexibility**: Mix and match different model implementations
3. **Checkpoint Management**: Resume from any stage
4. **Multiple Visualization Options**: Choose the best format for your needs
5. **Functional Programming**: No hidden state, clear data flow

## Next Steps

Now that you've run your first analysis with Kura, you can:

- [Learn about configuration options](configuration.md) to customize Kura
- Explore [core concepts](../core-concepts/overview.md) to understand how Kura works
- Check out the [API Reference](../api/index.md) for detailed documentation
