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

## Installation

```bash
uv pip install kura
```

## Quick Start

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
