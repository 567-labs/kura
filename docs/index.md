# Kura: Procedural API for Chat Data Analysis

![Kura Architecture](assets/images/kura-architecture.png)

[![PyPI Downloads](https://img.shields.io/pypi/dm/kura?style=flat-square&logo=pypi&logoColor=white)](https://pypi.org/project/kura/)
[![GitHub Stars](https://img.shields.io/github/stars/567-labs/kura?style=flat-square&logo=github)](https://github.com/567-labs/kura/stargazers)
[![Documentation](https://img.shields.io/badge/docs-available-brightgreen?style=flat-square&logo=gitbook&logoColor=white)](https://567-labs.github.io/kura/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/pypi/pyversions/kura?style=flat-square&logo=python&logoColor=white)](https://pypi.org/project/kura/)
[![PyPI Version](https://img.shields.io/pypi/v/kura?style=flat-square&logo=pypi&logoColor=white)](https://pypi.org/project/kura/)

Kura is an open-source library for understanding chat data through machine learning, inspired by [Anthropic's CLIO](https://www.anthropic.com/research/clio). It provides a functional, composable API for clustering conversations to discover patterns and insights.

## Why Analyze Conversation Data?

As AI assistants and chatbots become increasingly central to product experiences, understanding how users interact with these systems at scale becomes a critical challenge. Manually reviewing thousands of conversations is impractical, yet crucial patterns and user needs often remain hidden in this data.

Kura addresses this challenge by:

- **Revealing user intent patterns** that may not be obvious from individual conversations
- **Identifying common user needs** to prioritize feature development
- **Discovering edge cases and failures** that require attention
- **Tracking usage trends** over time as your product evolves
- **Informing prompt engineering** by highlighting successful and problematic interactions

## Features

- **Conversation Summarization**: Automatically generate concise task descriptions from conversations
- **Hierarchical Clustering**: Group similar conversations at multiple levels of granularity
- **Metadata Extraction**: Extract valuable context from conversations using LLMs
- **Custom Models**: Use your preferred embedding, summarization, and clustering methods
- **Checkpoint System**: Save and resume analysis sessions
- **Procedural API**: Functional approach with composable functions for maximum flexibility

## Installation

```bash
# Install from PyPI
pip install kura

# Or use uv for faster installation
uv pip install kura
```

## Quick Start

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

# Load conversations
conversations = Conversation.from_hf_dataset(
    "ivanleomk/synthetic-gemini-conversations",
    split="train"
)

# Set up models
summary_model = SummaryModel(cache_path="/cache/tmp/")
cluster_model = ClusterModel()
meta_cluster_model = MetaClusterModel(max_clusters=10)
dimensionality_model = HDBUMAP()

# Set up checkpoint manager
checkpoint_mgr = CheckpointManager("./checkpoints", enabled=False)

# Run pipeline with explicit steps
async def process_conversations():
    # Step 1: Generate summaries
    summaries = await summarise_conversations(
        conversations,
        model=summary_model,
        checkpoint_manager=checkpoint_mgr
    )

    # Step 2: Create base clusters
    clusters = await generate_base_clusters_from_conversation_summaries(
        summaries,
        model=cluster_model,
        checkpoint_manager=checkpoint_mgr
    )

    # Step 3: Build hierarchy
    meta_clusters = await reduce_clusters_from_base_clusters(
        clusters,
        model=meta_cluster_model,
        checkpoint_manager=checkpoint_mgr
    )

    # Step 4: Project to 2D
    projected = await reduce_dimensionality_from_clusters(
        meta_clusters,
        model=dimensionality_model,
        checkpoint_manager=checkpoint_mgr
    )

    return projected

# Execute the pipeline
results = asyncio.run(process_conversations())
visualise_pipeline_results(results, style="basic")
Clusters (190 conversations)
╠══ Generate SEO-optimized content for blogs and scripts (38 conversations)
║   ╠══ Assist in writing SEO-friendly blog posts (12 conversations)
║   ╚══ Help create SEO-driven marketing content (8 conversations)
╠══ Help analyze and visualize data with R and Tableau (25 conversations)
║   ╠══ Assist with data analysis and visualization in R (15 conversations)
║   ╚══ Troubleshoot sales data visualizations in Tableau (10 conversations)
... (and more clusters)
```

## Documentation

### Getting Started

- [Installation Guide](getting-started/installation.md)
- [Quickstart](getting-started/quickstart.md)

### Core Concepts

- [Conversations](core-concepts/conversations.md)
- [Embedding](core-concepts/embedding.md)
- [Clustering](core-concepts/clustering.md)
- [Summarization](core-concepts/summarization.md)
- [Meta-Clustering](core-concepts/meta-clustering.md)
- [Dimensionality Reduction](core-concepts/dimensionality-reduction.md)

### API Reference

- [Procedural API Documentation](api/index.md)

## About

Kura is under active development. If you face any issues or have suggestions, please feel free to [open an issue](https://github.com/567-labs/kura/issues) or a PR. For more details on the technical implementation, check out this [walkthrough of the code](https://ivanleo.com/blog/understanding-user-conversations).
