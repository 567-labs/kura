# Kura Documentation

## What is Kura?

> Kura is kindly sponsored by [Improving RAG](http://improvingrag.com). If you're wondering what goes on behind the scenes of any production RAG application, ImprovingRAG gives you a clear roadmap as to how to achieve it.

Kura is an open-source tool for understanding and visualizing chat data, inspired by [Anthropic's CLIO](https://www.anthropic.com/research/clio). It helps you discover patterns, trends, and insights from user conversations by applying machine learning techniques to cluster similar interactions.

By clustering similar conversations and providing intuitive visualizations, Kura transforms raw chat data into actionable insights without compromising user privacy. It uses language models like Gemini to understand conversation content and organize them into meaningful hierarchical groups, helping you focus on the specific features to prioritize or issues to fix.

I've written a [walkthrough of the code](https://ivanleo.com/blog/understanding-user-conversations) if you're interested in understanding the high level ideas.

## Why Analyze Conversation Data?

As AI assistants and chatbots become increasingly central to product experiences, understanding how users interact with these systems at scale becomes a critical challenge. Manually reviewing thousands of conversations is impractical, yet crucial patterns and user needs often remain hidden in this data.

Kura addresses this challenge by:

- **Revealing user intent patterns** that may not be obvious from individual conversations
- **Identifying common user needs** to prioritize feature development
- **Discovering edge cases and failures** that require attention
- **Tracking usage trends** over time as your product evolves
- **Informing prompt engineering** by highlighting successful and problematic interactions

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
- **Interactive Visualization**: Explore clusters through map, tree, and detail views
- **Metadata Extraction**: Extract valuable context from conversations using LLMs
- **Custom Models**: Use your preferred embedding, summarization, and clustering methods
- **Web Interface**: Intuitive UI for exploring and analyzing conversation clusters
- **CLI Tools**: Command-line interface for scripting and automation
- **Checkpoint System**: Save and resume analysis sessions

## Quick Start

> Kura requires python 3.9+ (Python 3.9 is specifically recommended due to UMAP dependency)

```python
from kura import Kura
from kura.types import Conversation
import asyncio

# Initialize Kura with checkpoint directory
kura = Kura(
    checkpoint_dir="./tutorial_checkpoints"
)

# Load sample data (190 synthetic programming conversations)
conversations = Conversation.from_hf_dataset(
    "ivanleomk/synthetic-gemini-conversations",
    split="train"
)
print(f"Loaded {len(conversations)} conversations successfully!")

# Process conversations through the clustering pipeline
asyncio.run(kura.cluster_conversations(conversations))
# Expected output:
# Generated 190 summaries
# Generated 19 base clusters
# Reduced to 29 meta clusters
# Generated 29 projected clusters

# Visualize the results
kura.visualise_clusters()
# Expected output: Hierarchical tree showing 10 root clusters
# with topics like:
# - Create engaging, SEO-optimized content for online platforms (40 conversations)
# - Help me visualize and analyze data across platforms (30 conversations)
# - Troubleshoot and implement authentication in web APIs (22 conversations)
# ... and more
```

## Architecture

Kura follows a modular, pipeline-based architecture:

1. **Data Loading**: Import conversations from various sources
2. **Summarization**: Generate concise descriptions of each conversation
3. **Embedding**: Convert text into vector representations
4. **Base Clustering**: Group similar summaries into initial clusters
5. **Meta-Clustering**: Create a hierarchical structure of clusters
6. **Dimensionality Reduction**: Project high-dimensional data for visualization
7. **Visualization**: Display clusters through web UI or CLI

### Core Components

- **`Kura`**: Main orchestrator for the entire pipeline
- **`OpenAIEmbeddingModel`**: Converts text to vector embeddings
- **`SummaryModel`**: Generates concise conversation summaries
- **`ClusterModel`**: Creates initial clusters from embeddings
- **`MetaClusterModel`**: Builds hierarchical structure from base clusters
- **`DimensionalityReduction`**: Projects data to 2D for visualization
- **`Conversation`**: Core data model for chat interactions

## Web Interface

Kura includes a React/TypeScript web interface with:

- **Cluster Map**: 2D visualization of conversation clusters
- **Cluster Tree**: Hierarchical view of cluster relationships
- **Cluster Details**: In-depth information about selected clusters
- **Conversation Dialog**: Examine individual conversations
- **Metadata Filtering**: Filter clusters based on extracted properties

Start the web interface with:

```bash
kura start-app
# Access at http://localhost:8000
```

## Next Steps

To explore more features, check out:
- [Installation Guide](getting-started/installation.md)
- [Quickstart Guide](getting-started/quickstart.md)
- [Core Concepts](core-concepts/overview.md)

## Technical Walkthrough

I've also recorded a technical deep dive into what Kura is and the ideas behind it if you'd rather watch than read.

<iframe width="560" height="315" src="https://www.youtube.com/embed/TPOP_jDiSVE?si=uvTond4LUwJGOn4F" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
