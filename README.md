# Kura: Procedural API for Chat Data Analysis

![Kura Architecture](./kura.png)

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

By clustering similar conversations and providing intuitive visualizations, Kura transforms raw chat data into actionable insights without compromising user privacy.

## Real-World Use Cases

- **Product Teams**: Understand how users engage with your AI assistant to identify opportunities for improvement
- **AI Research**: Analyze how different models respond to similar queries and detect systematic biases
- **Customer Support**: Identify common support themes and optimize response strategies
- **Content Creation**: Discover topics users are interested in to guide content development
- **Education**: Analyze student interactions with educational AI to improve learning experiences
- **UX Research**: Gain insights into user mental models and friction points

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
from kura.v1 import (
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
summary_model = SummaryModel()
cluster_model = ClusterModel()
meta_cluster_model = MetaClusterModel(max_clusters=10)
dimensionality_model = HDBUMAP()

# Set up checkpoint manager
checkpoint_mgr = CheckpointManager("./checkpoints", enabled=True)

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
```

## Key Design Principles

### Function-Based Architecture
The procedural API follows the principle of **functions orchestrate, models execute**:
- Each pipeline step is a pure function with explicit inputs/outputs
- No hidden state or side effects
- Works with any model implementing the required interface

### Polymorphism Through Interfaces
All functions work with heterogeneous models:
- `BaseSummaryModel` - OpenAI, vLLM, Hugging Face, local models
- `BaseClusterModel` - HDBSCAN, KMeans, custom algorithms
- `BaseMetaClusterModel` - Different hierarchical strategies
- `BaseDimensionalityReduction` - UMAP, t-SNE, PCA

### Keyword-Only Arguments
All functions use keyword-only arguments for clarity and maintainability.

## Loading Data

Kura supports multiple data sources:

### Claude Conversation History

```python
from kura.types import Conversation
conversations = Conversation.from_claude_conversation_dump("conversations.json")
```

### Hugging Face Datasets

```python
from kura.types import Conversation
conversations = Conversation.from_hf_dataset(
    "ivanleomk/synthetic-gemini-conversations",
    split="train"
)
```

> 💡 **Note:** This example uses a dataset of ~190 synthetic programming conversations that's structured for Kura. It contains technical discussions about web development frameworks, coding patterns, and software engineering that form natural clusters. The example loads and processes these conversations to create 29 hierarchical clusters across 10 root categories.

## Architecture

Kura follows a modular, pipeline-based architecture:

1. **Data Loading**: Import conversations from various sources
2. **Summarization**: Generate concise descriptions of each conversation
3. **Embedding**: Convert text into vector representations
4. **Base Clustering**: Group similar summaries into initial clusters
5. **Meta-Clustering**: Create a hierarchical structure of clusters
6. **Dimensionality Reduction**: Project high-dimensional data for visualization

### Core Components

- **`summarise_conversations`**: Generate conversation summaries using LLMs
- **`generate_base_clusters_from_conversation_summaries`**: Create initial clusters from embeddings
- **`reduce_clusters_from_base_clusters`**: Build hierarchical structure from base clusters
- **`reduce_dimensionality_from_clusters`**: Project data to 2D for visualization
- **`CheckpointManager`**: Save and resume analysis sessions
- **`Conversation`**: Core data model for chat interactions

## Checkpoints

Kura saves state between runs using checkpoint files:

- `conversations.json`: Raw conversation data
- `summaries.jsonl`: Summarized conversations
- `clusters.jsonl`: Base cluster data
- `meta_clusters.jsonl`: Hierarchical cluster data
- `dimensionality.jsonl`: Projected cluster data

Checkpoints are stored in the directory specified by `checkpoint_dir` (default: `./checkpoints`).

## Extending Kura

Kura is designed to be modular and extensible. You can create custom implementations of:

- Embedding models by extending `BaseEmbeddingModel`
- Summarization models by extending `BaseSummaryModel`
- Clustering algorithms by extending `BaseClusterModel`
- Meta-clustering methods by extending `BaseMetaClusterModel`
- Dimensionality reduction techniques by extending `BaseDimensionalityReduction`

## Documentation

- **Getting Started**
  - [Installation Guide](getting-started/installation.md)
  - [Tutorial: Procedural API](getting-started/tutorial-procedural-api.md)

- **Core Concepts**
  - [Conversations](core-concepts/conversations.md)
  - [Embedding](core-concepts/embedding.md)
  - [Clustering](core-concepts/clustering.md)
  - [Summarization](core-concepts/summarization.md)
  - [Meta-Clustering](core-concepts/meta-clustering.md)
  - [Dimensionality Reduction](core-concepts/dimensionality-reduction.md)

- **API Reference**
  - [Procedural API Documentation](api/index.md)

## Comparison with Similar Tools

| Feature | Kura | Traditional Analytics | Manual Review | Generic Clustering |
|---------|------|----------------------|--------------|-------------------|
| Semantic Understanding | ✅ Uses LLMs for deep understanding | ❌ Limited to keywords | ✅ Human understanding | ⚠️ Basic similarity only |
| Scalability | ✅ Handles thousands of conversations | ✅ Highly scalable | ❌ Time intensive | ✅ Works at scale |
| Visualization | ✅ Interactive UI | ⚠️ Basic charts | ❌ Manual effort | ⚠️ Generic plots |
| Hierarchy Discovery | ✅ Meta-clustering feature | ❌ Flat categories | ⚠️ Subjective grouping | ❌ Typically flat |
| Extensibility | ✅ Custom models and extractors | ⚠️ Limited customization | ✅ Flexible but manual | ⚠️ Some algorithms |
| Privacy | ✅ Self-hosted option | ⚠️ Often requires data sharing | ✅ Can be private | ✅ Can be private |

## Future Roadmap

Kura is actively evolving with plans to add:

- **Enhanced Topic Modeling**: More sophisticated detection of themes across conversations
- **Temporal Analysis**: Tracking how conversation patterns evolve over time
- **Advanced Visualizations**: Additional visual representations of conversation data
- **Data Connectors**: More integrations with popular conversation data sources
- **Multi-modal Support**: Analysis of conversations that include images and other media
- **Export Capabilities**: Enhanced formats for sharing and presenting findings

## Testing

To quickly test Kura and see it in action:

```bash
uv run python scripts/tutorial_procedural_api.py
```

Expected output:
```text
Loaded 190 conversations successfully!

============================================================
                  Conversation Processing
============================================================

Starting conversation clustering...
Step 1: Generating conversation summaries...
Generated 190 summaries
Step 2: Generating base clusters from summaries...
Generated 19 base clusters
Step 3: Reducing clusters hierarchically...
Reduced to 29 meta clusters
Step 4: Projecting clusters to 2D for visualization...
Generated 29 projected clusters

Pipeline complete! Generated 29 projected clusters!

Processing Summary:
  • Input conversations: 190
  • Final reduced clusters: 29
  • Final projected clusters: 29
  • Checkpoints saved to: ./tutorial_checkpoints
```

This will:
- Load 190 sample conversations from Hugging Face
- Process them through the complete pipeline
- Generate 29 hierarchical clusters organized into 10 root categories
- Save checkpoints to `./tutorial_checkpoints`

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, testing, and contribution guidelines.

## License

[MIT License](LICENSE)

## About

Kura is under active development. If you face any issues or have suggestions, please feel free to [open an issue](https://github.com/567-labs/kura/issues) or a PR. For more details on the technical implementation, check out this [walkthrough of the code](https://ivanleo.com/blog/understanding-user-conversations).
