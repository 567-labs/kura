# Kura: Chat Data Analysis and Visualization

![Kura Architecture](./kura.png)

Kura is an open-source tool for understanding and visualizing chat data, inspired by [Anthropic's CLIO](https://www.anthropic.com/research/clio). It helps you discover patterns, trends, and insights from user conversations by applying machine learning techniques to cluster similar interactions.

## Features

- **Conversation Summarization**: Automatically generate concise task descriptions from conversations
- **Hierarchical Clustering**: Group similar conversations at multiple levels of granularity
- **Interactive Visualization**: Explore clusters through map, tree, and detail views
- **Metadata Extraction**: Extract valuable context from conversations using LLMs
- **Custom Models**: Use your preferred embedding, summarization, and clustering methods
- **Web Interface**: Intuitive UI for exploring and analyzing conversation clusters
- **CLI Tools**: Command-line interface for scripting and automation
- **Checkpoint System**: Save and resume analysis sessions

## Installation

```bash
# Install from PyPI
pip install kura

# Or use uv for faster installation
uv pip install kura
```

## Quick Start

```python
from kura import Kura
from kura.embedding import OpenAIEmbeddingModel
from kura.summarisation import SummaryModel
from kura.dimensionality import DimensionalityReduction
from asyncio import run

# Initialize Kura with default components
kura = Kura(
    embedding_model=OpenAIEmbeddingModel(),
    summarisation_model=SummaryModel(),
    dimensionality_reduction=DimensionalityReduction(),
    max_clusters=10,
    checkpoint_dir="./checkpoints",
)

# Load conversations and run the clustering pipeline
kura.load_conversations("conversations.json")
run(kura.cluster_conversations())

# Visualize the results in the terminal
kura.visualise_clusters()

# Or start the web interface
# In your terminal: kura start-app
```

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

### Custom Conversations

```python
from kura.types import Conversation, Message
from datetime import datetime
from uuid import uuid4

conversations = [
    Conversation(
        messages=[
            Message(
                created_at=str(datetime.now()),
                role=message["role"],
                content=message["content"],
            )
            for message in raw_messages
        ],
        id=str(uuid4()),
        created_at=datetime.now(),
    )
]
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

## Working with Metadata

### LLM Extractors

Extract properties from conversations using LLM-powered functions:

```python
async def language_extractor(
    conversation: Conversation,
    sems: dict[str, asyncio.Semaphore],
    clients: dict[str, instructor.AsyncInstructor],
) -> ExtractedProperty:
    sem = sems.get("default")
    client = clients.get("default")
    
    async with sem:
        resp = await client.chat.completions.create(
            model="gemini-2.0-flash",
            messages=[
                {
                    "role": "system",
                    "content": "Extract the language of this conversation.",
                },
                {
                    "role": "user",
                    "content": "\n".join(
                        [f"{msg.role}: {msg.content}" for msg in conversation.messages]
                    ),
                },
            ],
            response_model=Language,
        )
        return ExtractedProperty(
            name="language_code",
            value=resp.language_code,
        )
```

### Conversation Metadata

Attach metadata directly when loading conversations:

```python
conversations = Conversation.from_hf_dataset(
    "allenai/WildChat-nontoxic",
    metadata_fn=lambda x: {
        "model": x["model"],
        "toxic": x["toxic"],
        "redacted": x["redacted"],
    },
)
```

## Checkpoints

Kura saves state between runs using checkpoint files:

- `conversations.json`: Raw conversation data
- `summaries.jsonl`: Summarized conversations
- `clusters.jsonl`: Base cluster data
- `meta_clusters.jsonl`: Hierarchical cluster data
- `dimensionality.jsonl`: Projected cluster data

Checkpoints are stored in the directory specified by `checkpoint_dir` (default: `./checkpoints`).

## CLI Commands

Run Kura's web interface:

```bash
# Start with default settings
kura start-app

# Use a custom checkpoint directory
kura start-app --dir ./my-checkpoints
```

## Extending Kura

Kura is designed to be modular and extensible. You can create custom implementations of:

- Embedding models by extending `BaseEmbeddingModel`
- Summarization models by extending `BaseSummaryModel`
- Clustering algorithms by extending `BaseClusterModel`
- Meta-clustering methods by extending `BaseMetaClusterModel`
- Dimensionality reduction techniques by extending `BaseDimensionalityReduction`

## Documentation

For more detailed documentation, run:

```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Serve documentation locally
mkdocs serve
# Access at http://localhost:8000
```

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, testing, and contribution guidelines.

## License

[MIT License](LICENSE)

## About

Kura is under active development. If you face any issues or have suggestions, please feel free to open an issue or a PR. For more details on the technical implementation, check out this [walkthrough of the code](https://ivanleo.com/blog/understanding-user-conversations).