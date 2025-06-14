# Kura: Procedural API for Chat Data Analysis

![Kura Architecture](assets/images/kura-architecture.png)

[![PyPI Downloads](https://img.shields.io/pypi/dm/kura?style=flat-square&logo=pypi&logoColor=white)](https://pypi.org/project/kura/)
[![GitHub Stars](https://img.shields.io/github/stars/567-labs/kura?style=flat-square&logo=github)](https://github.com/567-labs/kura/stargazers)
[![Documentation](https://img.shields.io/badge/docs-available-brightgreen?style=flat-square&logo=gitbook&logoColor=white)](https://567-labs.github.io/kura/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/pypi/pyversions/kura?style=flat-square&logo=python&logoColor=white)](https://pypi.org/project/kura/)
[![PyPI Version](https://img.shields.io/pypi/v/kura?style=flat-square&logo=pypi&logoColor=white)](https://pypi.org/project/kura/)

**Your AI assistant handles thousands of conversations daily. But do you know what users actually need?**

Kura is an open-source library for understanding chat data through machine learning, inspired by [Anthropic's CLIO](https://www.anthropic.com/research/clio). It automatically clusters conversations to reveal patterns, pain points, and opportunities hidden in your data.

## The Hidden Cost of Not Understanding Your Users

Every day, your AI assistant or chatbot has thousands of conversations. Within this data lies critical intelligence:

- **80% of support tickets** stem from 5 unclear features
- **Feature requests** repeated by hundreds of users differently
- **Revenue opportunities** from unmet needs
- **Critical failures** affecting user trust

### What Kura Does

Kura transforms unstructured conversation data into structured insights:

```
10,000 conversations → AI Analysis → 20 clear patterns
```

- **Intent Discovery**: Find what users actually want
- **Failure Detection**: Identify AI shortcomings early
- **Feature Prioritization**: See which missing features matter most
- **Semantic Clustering**: Group by meaning, not keywords
- **Privacy-First**: Analyze patterns without exposing conversations

## Key Features

- **Smart Summarization**: Convert conversations to task descriptions (with caching!)
- **Hierarchical Clustering**: Multi-level grouping
- **Metadata Extraction**: Language, sentiment, topics
- **Fully Extensible**: Bring your own models
- **Checkpoint System**: Never lose progress
- **Performance Optimized**: MiniBatch clustering, parallel processing
- **Web UI**: Interactive cluster visualization

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

# Set up models with new caching support!
summary_model = SummaryModel(
    enable_caching=True,  # NEW: 85x faster on re-runs!
    cache_dir="./.summary_cache"
)
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
visualise_pipeline_results(results, style="enhanced")

# Expected output:
Programming Assistance Clusters (190 conversations)
├── Data Analysis & Visualization (38 conversations)
│   ├── "Help me create R plots for statistical analysis"
│   ├── "Debug my Tableau dashboard performance issues"
│   └── "Convert Excel formulas to pandas operations"
├── Web Development (45 conversations)
│   ├── "Fix React component re-rendering issues"
│   ├── "Integrate Stripe API with Next.js"
│   └── "Make my CSS grid responsive on mobile"
└── ... (more clusters)

Performance: 21.9s first run → 2.1s with cache (10x faster!)
```

## Performance Features

### Smart Caching (New in v0.3.0+)

Kura now includes intelligent caching for expensive operations:

```python
# Enable caching for 85x faster development iterations
summary_model = SummaryModel(
    enable_caching=True,
    cache_dir="./.kura_cache",
    cache_ttl_days=7,  # Auto-expire old entries
)

# Cache automatically handles:
# - Content-based deduplication
# - Thread-safe operations
# - Automatic cleanup
# - Cross-session persistence
```

### Parallel Processing

```python
# Process multiple conversations simultaneously
summary_model = SummaryModel(
    max_concurrent_requests=100,  # Parallel API calls
    enable_caching=True,
)
```

### MiniBatch Clustering

```python
# Handle large datasets efficiently
from kura.k_means import MiniBatchKmeansClusteringMethod

clustering = MiniBatchKmeansClusteringMethod(
    batch_size=1000,  # Process in chunks
    clusters_per_group=10,
)
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

## Frequently Asked Questions

### How is Kura different from traditional analytics?

Traditional analytics focus on metrics (counts, rates, averages). Kura understands **meaning** - it knows that "How do I cancel?" and "I want to stop my subscription" are the same intent, even though they share no keywords.

### Can I use my own models?

Yes! Kura is model-agnostic. You can use:

- **OpenAI**: GPT-3.5, GPT-4
- **Anthropic**: Claude models
- **Local models**: Via Ollama, vLLM, or HuggingFace
- **Custom implementations**: Extend base classes

### How much data do I need?

- **Minimum**: 100 conversations for meaningful patterns
- **Recommended**: 1,000+ conversations for robust clustering
- **Optimal**: 10,000+ conversations for detailed insights

### Is my data secure?

- **Self-hosted option**: Run entirely on your infrastructure
- **No data retention**: API providers don't store your conversations
- **Privacy-first**: Analyze patterns without exposing individual chats
- **Configurable**: Use local models for complete data isolation

### What's the typical workflow?

1. **Export** conversations from your platform
2. **Load** into Kura using appropriate loader
3. **Process** through the pipeline (summarize → cluster → visualize)
4. **Analyze** results in web UI or export findings
5. **Act** on insights to improve your product

### How do I handle non-English conversations?

Kura works with any language supported by your chosen model:

```python
# Use a multilingual model
summary_model = SummaryModel(
    model="gpt-4",  # Supports 90+ languages
    custom_instructions="Respond in the same language as the input"
)
```

### Can I integrate Kura into my application?

Yes! Kura is designed as a library:

```python
# Use in your async application
from kura import summarise_conversations

async def analyze_user_feedback(conversations):
    summaries = await summarise_conversations(
        conversations,
        model=summary_model
    )
    return summaries
```

## About

Kura is under active development. If you face any issues or have suggestions, please feel free to [open an issue](https://github.com/567-labs/kura/issues) or a PR. For more details on the technical implementation, check out this [walkthrough of the code](https://ivanleo.com/blog/understanding-user-conversations).
