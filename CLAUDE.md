# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Python Environment Setup

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package in development mode with dev dependencies
pip install -e ".[dev]"
```

### Running Tests

```bash
# Run all tests
pytest

# Run a specific test file
pytest tests/test_meta_cluster.py

# Run a specific test
pytest tests/test_meta_cluster.py::test_cluster_label_exact_match
```

### Type Checking

```bash
# Run type checking
pyright
```

### Documentation

```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Serve documentation locally
mkdocs serve
```

### UI Development

```bash
# Navigate to UI directory
cd ui

# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Lint code
npm run lint
```

### Running the Application

```bash
# Start the Kura web server (implemented in kura/cli/cli.py and kura/cli/server.py)
kura start-app

# Start with a custom checkpoint directory
kura start-app --dir ./my-checkpoints

# Start with HuggingFace datasets checkpoints (recommended for large datasets)
kura start-app --checkpoint-format hf-dataset
```

### Checkpoint Management

```bash
# Analyze existing JSONL checkpoints and estimate migration benefits
kura analyze-checkpoints ./checkpoints

# Migrate JSONL checkpoints to HuggingFace datasets format
kura migrate-checkpoints ./old_checkpoints ./new_hf_checkpoints

# Migrate with HuggingFace Hub upload and compression
kura migrate-checkpoints ./old_checkpoints ./new_hf_checkpoints \
    --hub-repo my-username/kura-analysis \
    --hub-token $HF_TOKEN \
    --compression gzip
```

## Architecture Overview

Kura is a tool for analyzing and visualizing chat data, built on the same ideas as Anthropic's CLIO. It uses machine learning techniques to understand user conversations by clustering them into meaningful groups.

### Two API Approaches

Kura offers two APIs for different use cases:

1. **Class-Based API** (`kura/kura.py`): The original API with a single `Kura` class that orchestrates the entire pipeline
2. **Procedural API** (`kura/v1/`): A functional approach with composable functions for maximum flexibility

### Core Components

1. **Summarisation Model** (`kura/summarisation.py`): Takes user conversations and summarizes them into task descriptions
2. **Embedding Model** (`kura/embedding.py`): Converts text into vector representations (embeddings)
3. **Clustering Model** (`kura/cluster.py`): Groups summaries into clusters based on embeddings
4. **Meta Clustering Model** (`kura/meta_cluster.py`): Further groups clusters into a hierarchical structure (Note: `max_clusters` parameter now lives here, not in the main Kura class)
5. **Dimensionality Reduction** (`kura/dimensionality.py`): Reduces high-dimensional embeddings for visualization

### Data Flow

1. Raw conversations are loaded
2. Conversations are summarized
3. Summaries are embedded and clustered
4. Base clusters are reduced to meta-clusters
5. Dimensionality reduction is applied for visualization
6. Results are saved as checkpoints for persistence

### Key Classes

- `Kura` (`kura/kura.py`): Main class that orchestrates the entire pipeline
- `BaseEmbeddingModel` / `OpenAIEmbeddingModel` (`kura/embedding.py`): Handle text embedding
- `BaseSummaryModel` / `SummaryModel` (`kura/summarisation.py`): Summarize conversations
- `BaseClusterModel` / `ClusterModel` (`kura/cluster.py`): Create initial clusters
- `BaseMetaClusterModel` / `MetaClusterModel` (`kura/meta_cluster.py`): Reduce clusters into hierarchical groups
- `BaseDimensionalityReduction` / `HDBUMAP` (`kura/dimensionality.py`): Reduce dimensions for visualization
- `Conversation` (`kura/types/conversation.py`): Core data model for user conversations

### UI Components

The project includes a React/TypeScript frontend for visualizing the clusters, with components for:
- Displaying cluster maps (`ui/src/components/cluster-map.tsx`)
- Showing cluster details (`ui/src/components/cluster-details.tsx`)
- Visualizing cluster hierarchies (`ui/src/components/cluster-tree.tsx`)
- Handling conversation uploads (`ui/src/components/upload-form.tsx`)
- Displaying individual conversations (`ui/src/components/conversation-dialog.tsx`)

### Extensibility

The system is designed to be modular, allowing custom implementations of:
- Embedding models
- Summarization models
- Clustering algorithms
- Dimensionality reduction techniques

## Working with Metadata

Kura supports two types of metadata for enriching conversation analysis:

### 1. LLM Extractors
Custom metadata can be extracted from conversations using LLM-powered extractors (implemented in `kura/summarisation.py`). These functions run on raw conversations to identify properties like:
- Language detection
- Sentiment analysis
- Topic identification
- Custom metrics

Example of creating a custom extractor:
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

### 2. Conversation Metadata
Metadata can be directly attached to conversation objects when loading data (implemented in `kura/types/conversation.py`):
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

## Loading Data

Kura supports multiple data sources (implementations in `kura/types/conversation.py`):

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
For custom data formats, create Conversation objects directly:
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

## Checkpoints

Kura supports two checkpoint systems for storing intermediate pipeline results:

### 1. JSONL Checkpoints (Legacy)

The traditional system uses JSONL files (checkpoint handling in `kura/kura.py`):
- `conversations.json`: Raw conversation data
- `summaries.jsonl`: Summarized conversations
- `clusters.jsonl`: Base cluster data
- `meta_clusters.jsonl`: Hierarchical cluster data
- `dimensionality.jsonl`: Projected cluster data for visualization

Checkpoints are stored in the directory specified by the `checkpoint_dir` parameter (default: `./checkpoints`).

### 2. HuggingFace Datasets Checkpoints (Recommended)

The new system uses HuggingFace datasets for better performance and scalability:

**Benefits:**
- **Memory Efficiency**: 10-100x better for large datasets using memory-mapped files
- **Performance**: Significantly faster loading with built-in optimization
- **Compression**: 50-80% smaller storage footprint with built-in compression
- **Streaming**: Process datasets larger than available RAM
- **Querying**: Filter and query without loading entire dataset
- **Versioning**: Built-in version control via HuggingFace Hub
- **Collaboration**: Easy sharing via HuggingFace Hub
- **Schema Validation**: Automatic data structure validation

**Usage with Procedural API:**
```python
from kura.v1 import create_hf_checkpoint_manager

# Create HF datasets checkpoint manager
checkpoint_mgr = create_hf_checkpoint_manager(
    checkpoint_dir="./hf_checkpoints",
    hub_repo="my-username/kura-analysis",  # Optional: cloud backup
    compression="gzip",                    # Built-in compression
    streaming=True                         # Handle large datasets
)

# Use with any pipeline function
summaries = await summarise_conversations(
    conversations,
    model=summary_model,
    checkpoint_manager=checkpoint_mgr
)
```

**Advanced Features:**
```python
# Filter checkpoints efficiently
large_clusters = checkpoint_mgr.filter_checkpoint(
    "clusters",
    lambda x: len(x["chat_ids"]) > 100,  # Only large clusters
    Cluster
)

# Get checkpoint statistics
info = checkpoint_mgr.get_checkpoint_info("summaries")
print(f"Size: {info['size_bytes']} bytes, Rows: {info['num_rows']}")
```

**Format Selection:**
Choose checkpoint format based on needs:
- **JSONL**: Small datasets (< 10MB), quick prototyping, legacy compatibility
- **HF Datasets**: Large datasets (> 100MB), production deployments, collaboration

## Visualization

Kura includes visualization tools:

### CLI Visualization
```python
# Tree visualization implemented in kura/kura.py
kura.visualise_clusters()
```

### Web Server
```bash
# Web server implemented in kura/cli/server.py
kura start-app
# Access at http://localhost:8000
```

The web interface provides:
- Interactive cluster map
- Cluster hierarchy tree
- Cluster details panel
- Conversation preview
- Metadata filtering

## Procedural API (v1)

The procedural API in `kura/v1/` provides a functional approach to the pipeline:

### Key Functions
- `summarise_conversations(conversations, *, model, checkpoint_manager=None)` - Generate summaries
- `generate_base_clusters_from_conversation_summaries(summaries, *, model, checkpoint_manager=None)` - Create initial clusters
- `reduce_clusters_from_base_clusters(clusters, *, model, checkpoint_manager=None)` - Build hierarchy
- `reduce_dimensionality_from_clusters(clusters, *, model, checkpoint_manager=None)` - Project to 2D

### Example Usage
```python
from kura import (
    summarise_conversations,
    generate_base_clusters_from_conversation_summaries,
    reduce_clusters_from_base_clusters,
    reduce_dimensionality_from_clusters,
    CheckpointManager
)

# Run pipeline with explicit steps
checkpoint_mgr = CheckpointManager("./checkpoints", enabled=True)

summaries = await summarise_conversations(
    conversations,
    model=summary_model,
    checkpoint_manager=checkpoint_mgr
)

clusters = await generate_base_clusters_from_conversation_summaries(
    summaries,
    model=cluster_model,
    checkpoint_manager=checkpoint_mgr
)
# ... continue with remaining steps
```

### Benefits
- Fine-grained control over each step
- Easy to skip or reorder steps
- Support for heterogeneous models (OpenAI, vLLM, Hugging Face, etc.)
- Functional programming style with no hidden state
- All functions use keyword-only arguments for clarity