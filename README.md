# Kura: Procedural API for Chat Data Analysis

![Kura Architecture](./kura.png)

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

- **80% of support tickets** might stem from the same 5 unclear features
- **Key feature requests** repeated by hundreds of users in different ways
- **Revenue opportunities** from unmet needs you didn't know existed
- **Critical failures** affecting user trust that go unreported

Manually reviewing conversations doesn't scale. Traditional analytics miss semantic meaning. **Kura bridges this gap.**

## What Kura Does

Kura transforms unstructured conversation data into structured insights:

```
10,000 conversations → AI Analysis → 20 clear patterns
```

- **Automatic Intent Discovery**: Find what users actually want (not what they say)
- **Failure Pattern Detection**: Identify where your AI falls short before users complain
- **Feature Priority Insights**: See which missing features impact the most users
- **Semantic Clustering**: Group by meaning, not keywords
- **Privacy-First Design**: Analyze patterns without exposing individual conversations

## Real-World Impact

### E-commerce Support Bot

**Challenge**: 50,000 weekly conversations, unknown pain points
**Discovery**: 35% of conversations about shipping clustered into 3 issues
**Result**: Fixed root causes, reduced support volume by 40%

### Developer Documentation Assistant

**Challenge**: Users struggling but not reporting specific issues
**Discovery**: 2,000+ conversations revealed 5 consistently confusing APIs
**Result**: Targeted doc improvements, 60% reduction in those queries

### SaaS Onboarding Bot

**Challenge**: 30% of trials not converting, unclear why
**Discovery**: Clustering revealed 3 missing integration requests
**Result**: Built integrations, trial conversion increased 18%

## Installation

```bash
uv pip install kura
```

## When to Use Kura

**Kura is perfect when you have:**

- 100+ conversations to analyze (scales to millions)
- A need to understand user patterns, not individual conversations
- Unstructured conversation data from chatbots, support systems, or AI assistants
- Questions like "What are users struggling with?" or "What features are they requesting?"

**Kura might not be the best fit if:**

- You have fewer than 100 conversations (manual review might be faster)
- You need real-time analysis (Kura is designed for batch processing)
- You only need keyword-based search (use traditional search tools instead)
- You require conversation-level sentiment analysis (Kura focuses on patterns)

## Common Use Cases

### Product Teams

- **Feature Discovery**: Find the features users ask for in their own words
- **Pain Point Analysis**: Identify friction in user journeys
- **Roadmap Prioritization**: Quantify impact of potential improvements

### Customer Success

- **Support Deflection**: Find common issues to create better docs/FAQs
- **Escalation Patterns**: Identify conversations that lead to churn
- **Success Patterns**: Discover what makes users successful

### AI/ML Teams

- **Prompt Engineering**: Find where prompts fail or confuse users
- **Model Evaluation**: Understand model performance beyond metrics
- **Training Data**: Identify gaps in model knowledge

### Analytics Teams

- **Behavioral Insights**: Understand user segments by conversation patterns
- **Trend Analysis**: Track how user needs evolve over time
- **ROI Measurement**: Connect conversation patterns to business outcomes

## Quick Start

### From Zero to Insights in 5 Minutes

```python
import asyncio
from rich.console import Console
from kura import (
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
    # Initialize models
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
    visualise_pipeline_results(projected_clusters, style="enhanced")


if __name__ == "__main__":
    asyncio.run(main())
```

### What This Example Does

1. **Loads** 190 real programming conversations from Hugging Face
2. **Summarizes** each conversation into a concise task description (with caching!)
3. **Clusters** similar conversations using MiniBatch K-means for speed
4. **Organizes** clusters into a hierarchy for easy navigation
5. **Visualizes** the results in your terminal

### Expected Output

```text
Programming Assistance (190 conversations)
├── Data Analysis & Visualization (38 conversations)
│   ├── R Programming for statistical analysis (12 conversations)
│   ├── Tableau dashboard creation (10 conversations)
│   └── Python data manipulation with pandas (16 conversations)
├── Web Development (45 conversations)
│   ├── React component development (20 conversations)
│   ├── API integration issues (15 conversations)
│   └── CSS styling and responsive design (10 conversations)
├── Machine Learning (32 conversations)
│   ├── Model training with TensorFlow (18 conversations)
│   └── Data preprocessing challenges (14 conversations)
└── ... (more clusters)

Total processing time: 21.9s (2.1s with cache!)
Checkpoints saved to: ./checkpoints/
```

## Performance Optimization Guide

### For Large Datasets (10k+ conversations)

1. **Use MiniBatch K-means clustering**:

```python
from kura.k_means import MiniBatchKmeansClusteringMethod

clustering = MiniBatchKmeansClusteringMethod(
    batch_size=1000,      # Process in chunks
    max_iter=100,         # Limit iterations
    clusters_per_group=15 # Balance quality vs speed
)
```

2. **Enable streaming with HuggingFace checkpoints**:

```python
checkpoint_mgr = HFDatasetCheckpointManager(
    "./checkpoints",
    streaming=True,  # Don't load everything into memory
    compression="gzip"  # 50-80% smaller files
)
```

3. **Optimize API concurrency**:

```python
# Find the sweet spot for your API limits
summary_model = SummaryModel(
    max_concurrent_requests=50,
)
```

### Speed Optimization Tips

| Optimization         | Impact                    | When to Use                            |
| -------------------- | ------------------------- | -------------------------------------- |
| Enable caching       | 85x faster re-runs        | Development, iteration                 |
| MiniBatch clustering | 3-5x faster               | Datasets > 5k conversations            |
| Reduce max_clusters  | 2x faster meta-clustering | When fewer top-level groups acceptable |
| Use local models     | No API latency            | When quality tradeoff acceptable       |
| Parallel checkpoints | 40% faster I/O            | Always (enabled by default)            |

### Memory Optimization

```python
# Process large datasets in batches
async def process_large_dataset(conversations, batch_size=1000):
    all_summaries = []

    for i in range(0, len(conversations), batch_size):
        batch = conversations[i:i+batch_size]
        summaries = await summarise_conversations(
            batch,
            model=summary_model,
            checkpoint_manager=checkpoint_mgr
        )
        all_summaries.extend(summaries)

        # Free memory between batches
        import gc
        gc.collect()

    return all_summaries
```

### Cost Optimization

1. **Use cheaper models for initial exploration**:

```python
# Start with GPT-3.5 for exploration
exploration_model = SummaryModel(model="gpt-3.5-turbo")

# Switch to GPT-4 for production
production_model = SummaryModel(model="gpt-4")
```

2. **Cache aggressively during development**:

```python
summary_model = SummaryModel(
    enable_caching=True,
    cache_ttl_days=30  # Long TTL for stable datasets
)
```

## Performance & Caching

### Summary Caching (New!)

Kura now supports intelligent caching of conversation summaries:

```python
# Enable caching to dramatically speed up iterative development
summary_model = SummaryModel(
    enable_caching=True,
    cache_dir="./.summary_cache",
    cache_ttl_days=7  # Cache entries expire after 7 days
)

# First run: ~8.5s for 190 conversations
# Subsequent runs: ~0.1s (85x faster!)
```

**Benefits:**

- 85x faster on cached runs
- Persistent across sessions
- Content-based cache keys
- Automatic cache expiration
- Thread-safe implementation

## Key Design Principles

Kura follows a function-based architecture where pipeline functions orchestrate the execution while models handle the core logic. Each function is designed with explicit inputs/outputs and no hidden state, working with any model that implements the required interface. The system supports various model types through polymorphic interfaces - from OpenAI to local models for summarization, different clustering algorithms, and various dimensionality reduction techniques.

Data can be loaded from multiple sources including Claude conversation history (`Conversation.from_claude_conversation_dump()`) and Hugging Face datasets (`Conversation.from_hf_dataset()`). The example uses a dataset of 190 synthetic programming conversations that form natural clusters across technical topics.

The pipeline architecture processes data through sequential stages: loading, summarization, embedding, base clustering, meta-clustering, and dimensionality reduction. All progress is automatically saved using checkpoints, and the system can be extended by implementing custom versions of any component model.

## Documentation

- **Getting Started**

  - [Installation Guide](https://567-labs.github.io/kura/getting-started/installation/)
  - [Quickstart Guide](https://567-labs.github.io/kura/getting-started/quickstart/)

- **Core Concepts**

  - [Conversations](https://567-labs.github.io/kura/core-concepts/conversations/)
  - [Embedding](https://567-labs.github.io/kura/core-concepts/embedding/)
  - [Clustering](https://567-labs.github.io/kura/core-concepts/clustering/)
  - [Summarization](https://567-labs.github.io/kura/core-concepts/summarization/)
  - [Meta-Clustering](https://567-labs.github.io/kura/core-concepts/meta-clustering/)
  - [Dimensionality Reduction](https://567-labs.github.io/kura/core-concepts/dimensionality-reduction/)

- **API Reference**
  - [Procedural API Documentation](https://567-labs.github.io/kura/api/)

## Checkpoint System

Kura provides three checkpoint managers for different use cases:

| Checkpoint Manager             | Format      | Dependencies      | File Size   | Use Case                               |
| ------------------------------ | ----------- | ----------------- | ----------- | -------------------------------------- |
| **JSONLCheckpointManager**     | JSON Lines  | None              | Baseline    | Development, debugging, small datasets |
| **ParquetCheckpointManager**   | Parquet     | PyArrow           | 50% smaller | Production workflows, analytics        |
| **HFDatasetCheckpointManager** | HF Datasets | datasets, PyArrow | 7% smaller  | Large-scale ML, cloud workflows        |

### Checkpoint Performance (190 conversations)

Based on tutorial benchmarks:

```text
JSONL: 200KB total storage
PARQUET: 100KB total storage (50% space savings)
HF: 186KB total storage (7% space savings)
```

- **JSONL**: Human-readable, universal compatibility, no dependencies
- **Parquet**: Best compression, fastest analytical queries, type safety
- **HuggingFace**: Streaming support, cloud sync, versioning, advanced features

## Comparison with Similar Tools

| Feature                | Kura                                  | Traditional Analytics          | Manual Review          | Generic Clustering       |
| ---------------------- | ------------------------------------- | ------------------------------ | ---------------------- | ------------------------ |
| Semantic Understanding | ✅ Uses LLMs for deep understanding   | ❌ Limited to keywords         | ✅ Human understanding | ⚠️ Basic similarity only |
| Scalability            | ✅ Handles thousands of conversations | ✅ Highly scalable             | ❌ Time intensive      | ✅ Works at scale        |
| Visualization          | ✅ Interactive UI                     | ⚠️ Basic charts                | ❌ Manual effort       | ⚠️ Generic plots         |
| Hierarchy Discovery    | ✅ Meta-clustering feature            | ❌ Flat categories             | ⚠️ Subjective grouping | ❌ Typically flat        |
| Extensibility          | ✅ Custom models and extractors       | ⚠️ Limited customization       | ✅ Flexible but manual | ⚠️ Some algorithms       |
| Privacy                | ✅ Self-hosted option                 | ⚠️ Often requires data sharing | ✅ Can be private      | ✅ Can be private        |

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

This script tests all three checkpoint managers and provides timing comparisons. Expected output:

```text
Loaded 190 conversations successfully!

Saved 190 conversations to ./tutorial_checkpoints/conversations.json

Running with HFDatasetCheckpointManager
Step 1: Generating summaries with checkpoints...
Generated 190 summaries using checkpoints
Step 2: Generating clusters with checkpoints...
Generated 19 clusters using checkpoints
Step 3: Meta clustering with checkpoints...
Reduced to 29 meta clusters using checkpoints
Step 4: Dimensionality reduction with checkpoints...
Generated 29 projected clusters using HFDatasetCheckpointManager

Running with ParquetCheckpointManager
Step 1: Generating summaries with checkpoints...
Generated 190 summaries using checkpoints
Step 2: Generating clusters with checkpoints...
Generated 19 clusters using checkpoints
Step 3: Meta clustering with checkpoints...
Reduced to 29 meta clusters using checkpoints
Step 4: Dimensionality reduction with checkpoints...
Generated 29 projected clusters using ParquetCheckpointManager

Running with JSONLCheckpointManager
Step 1: Generating summaries with checkpoints...
Generated 190 summaries using checkpoints
Step 2: Generating clusters with checkpoints...
Generated 19 clusters using checkpoints
Step 3: Meta clustering with checkpoints...
Reduced to 29 meta clusters using checkpoints
Step 4: Dimensionality reduction with checkpoints...
Generated 29 projected clusters using JSONLCheckpointManager

============================================================
                    TIMING SUMMARY
============================================================
Loading sample conversations               1.23s ( 5.2%)
Saving conversations to JSON               0.45s ( 1.9%)
HFDatasetCheckpointManager - Summarization 8.45s (35.8%)
HFDatasetCheckpointManager - Clustering    6.78s (28.7%)
HFDatasetCheckpointManager - Meta clustering 4.32s (18.3%)
HFDatasetCheckpointManager - Dimensionality 2.34s (9.9%)
------------------------------------------------------------
Total Time                               23.57s
============================================================
```

This will:

- Load 190 sample conversations from Hugging Face
- Process them through the complete pipeline with each checkpoint manager
- Compare timing and storage efficiency across formats
- Generate 29 hierarchical clusters organized into categories
- Save checkpoints to `./tutorial_checkpoints/` with subfolders for each format

## Troubleshooting

### Common Issues & Solutions

#### "Rate limit exceeded" errors

**Problem**: Getting rate limit errors from OpenAI/Anthropic
**Solution**: Reduce concurrent requests:

```python
summary_model = SummaryModel(max_concurrent_requests=10)  # Lower from default 100
```

#### Out of memory with large datasets

**Problem**: Memory errors when processing 10k+ conversations
**Solutions**:

1. Use HuggingFace checkpoint manager for streaming:

```python
checkpoint_mgr = HFDatasetCheckpointManager(
    "./checkpoints",
    streaming=True  # Enable streaming mode
)
```

2. Process in batches:

```python
# Process conversations in chunks
for i in range(0, len(conversations), 1000):
    batch = conversations[i:i+1000]
    summaries = await summarise_conversations(batch, model=summary_model)
```

#### "No module named 'instructor'" error

**Problem**: Missing required dependencies
**Solution**: Install with all dependencies:

```bash
uv pip install kura[all]
# or
pip install kura[all]
```

#### Clusters seem random or poor quality

**Problem**: Clusters don't make semantic sense
**Solutions**:

1. Ensure sufficient data (100+ conversations minimum)
2. Adjust clustering parameters:

```python
clustering = MiniBatchKmeansClusteringMethod(
    clusters_per_group=5,  # Fewer items per cluster for better cohesion
    random_state=42  # Set seed for reproducibility
)
```

3. Try different embedding models:

```python
from kura.embedding import OpenAIEmbeddingModel
embedding_model = OpenAIEmbeddingModel(model="text-embedding-3-large")
```

#### Cache not working

**Problem**: Summaries regenerating despite cache enabled
**Solution**: Check cache directory permissions:

```python
import os
cache_dir = "./.summary_cache"
os.makedirs(cache_dir, exist_ok=True)
# Ensure write permissions
```

#### "Connection refused" for web UI

**Problem**: Can't access web interface
**Solutions**:

1. Check if another process is using port 8000:

```bash
lsof -i :8000  # macOS/Linux
netstat -ano | findstr :8000  # Windows
```

2. Use a different port:

```bash
kura start-app --port 8080
```

### Getting Help

1. **Check existing issues**: [GitHub Issues](https://github.com/567-labs/kura/issues)
2. **Join discussions**: [GitHub Discussions](https://github.com/567-labs/kura/discussions)
3. **Read the docs**: [Full Documentation](https://567-labs.github.io/kura/)
4. **Debug mode**: Set environment variable for verbose logging:

```bash
export KURA_LOG_LEVEL=DEBUG
```

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, testing, and contribution guidelines.

## License

[MIT License](LICENSE)

## About

Kura is under active development. If you face any issues or have suggestions, please feel free to [open an issue](https://github.com/567-labs/kura/issues) or a PR. For more details on the technical implementation, check out this [walkthrough of the code](https://ivanleo.com/blog/understanding-user-conversations).
