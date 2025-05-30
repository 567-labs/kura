# Quickstart Guide

This guide will help you get started with Kura quickly. We'll cover the basic workflow of analyzing a dataset using Kura's default settings.

## Prerequisites

Before you begin, make sure you have:

1. [Installed Kura](installation.md)
2. Set up your API key for the default Gemini model:
   ```bash
   export GOOGLE_API_KEY=your_api_key_here
   ```

## Basic Workflow

Kura's basic workflow consists of:

1. Loading conversational data
2. Processing the data through summarization, embedding, and clustering
3. Visualizing the results

Let's walk through each step.

## Sample Code

Here's a complete example to get you started with Kura using a sample dataset:

```python
from kura import (
    summarise_conversations,
    generate_base_clusters_from_conversation_summaries,
    reduce_clusters_from_base_clusters,
    reduce_dimensionality_from_clusters,
    CheckpointManager
)
from kura.types import Conversation
from kura.summarisation import SummaryModel
from kura.cluster import ClusterModel
from kura.meta_cluster import MetaClusterModel
from kura.dimensionality import HDBUMAP
import asyncio

# Load sample conversations from Hugging Face
# This loads 190 synthetic programming conversations
conversations = Conversation.from_hf_dataset(
    "ivanleomk/synthetic-gemini-conversations",
    split="train"
)
# Expected output: "Loaded 190 conversations successfully!"

async def main():
    # Initialize models and checkpoint manager
    summary_model = SummaryModel()
    cluster_model = ClusterModel()
    meta_cluster_model = MetaClusterModel()
    dimensionality_model = HDBUMAP()
    checkpoint_manager = CheckpointManager("./tutorial_checkpoints", enabled=True)
    
    # Run the clustering pipeline step by step
    # 1. Generate conversation summaries
    summaries = await summarise_conversations(
        conversations, 
        model=summary_model, 
        checkpoint_manager=checkpoint_manager
    )
    
    # 2. Create base clusters from summaries
    clusters = await generate_base_clusters_from_conversation_summaries(
        summaries, 
        model=cluster_model, 
        checkpoint_manager=checkpoint_manager
    )
    
    # 3. Reduce clusters hierarchically
    reduced_clusters = await reduce_clusters_from_base_clusters(
        clusters, 
        model=meta_cluster_model, 
        checkpoint_manager=checkpoint_manager
    )
    
    # 4. Project clusters to 2D for visualization
    result = await reduce_dimensionality_from_clusters(
        reduced_clusters, 
        model=dimensionality_model, 
        checkpoint_manager=checkpoint_manager
    )

asyncio.run(main())

# Visualize the results in the terminal
from kura.visualization import visualise_clusters_rich
visualise_clusters_rich(result)
# Expected output: Hierarchical tree showing clusters with topics like:
# - Create engaging, SEO-optimized content for online platforms (40 conversations)
# - Help me visualize and analyze data across platforms (30 conversations)
# - Troubleshoot and implement authentication in web APIs (22 conversations)
# ... and more
```

This will:

1. Initialize models and checkpoint manager for saving results
2. Load 190 synthetic programming conversations from Hugging Face  
3. Process them through the complete analysis pipeline step by step
4. Generate hierarchical clusters organized into categories
5. Display the hierarchical clustering results in the terminal

### Expected Visualization Output

When you run `kura.visualise_clusters()`, you'll see a hierarchical tree view like this:

```
Clusters (190 conversations)
╠══ Create engaging, SEO-optimized content for online platforms (40 conversations)
║   ╠══ Create SEO-focused marketing content for products (8 conversations)
║   ╠══ Create engaging YouTube video scripts for tutorials (20 conversations)
║   ╚══ Assist in writing engaging SEO-friendly blog posts (12 conversations)
╠══ Help me visualize and analyze data across platforms (30 conversations)
║   ╠══ Assist with R data analysis and visualization issues (9 conversations)
║   ╠══ Assist with data analysis and visualization in Python (12 conversations)
║   ╚══ Help me visualize sales data in Tableau (9 conversations)
╠══ Troubleshoot and implement authentication in web APIs (22 conversations)
║   ╠══ Guide on implementing JWT authentication in Spring Boot (10 conversations)
║   ╠══ Troubleshoot API authentication issues in a Flutter app (2 conversations)
║   ╚══ Assist in troubleshooting Django REST API issues (10 conversations)
╠══ Improve performance of ETL and real-time data pipelines (21 conversations)
║   ╠══ Optimize ETL pipelines for performance and quality (9 conversations)
║   ╚══ Optimize real-time data pipelines using Spark and Kafka (12 conversations)
... (and more clusters)
```

## Using the Web Interface

For a more interactive experience, Kura includes a web interface:

```bash
# Start with default checkpoint directory
kura start-app

# Or use a custom checkpoint directory
kura start-app --dir ./tutorial_checkpoints
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

## Next Steps

Now that you've run your first analysis with Kura, you can:

- [Learn about configuration options](configuration.md) to customize Kura
- Explore [core concepts](../core-concepts/overview.md) to understand how Kura works
- Try the [Procedural API Tutorial](tutorial-procedural-api.md) for a more flexible approach
- Check out the [API Reference](../api/index.md) for detailed documentation
