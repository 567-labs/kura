# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Guidelines

### Workflow Rules
- Always make pull requests for code changes
- Never push directly to main branch
- Create feature branches for new functionality
- Make sure all files are properly completed before committing
- Don't use `git add .` or `git add --all` - be selective about what you stage

### Writing Standards
- Write at a 9th-grade reading level, always
- Use clear, simple language in documentation and comments
- Avoid technical jargon unless necessary
- Don't use emojis in commit messages or documentation

### Python Development
- Always use `uv` instead of `pip` for package management
- Follow existing code style and patterns in the project
- Add type hints to all new functions and methods
- Write docstrings for all public functions and classes

### Optional Dependencies
When working with optional dependencies in Kura (like `rich`, `pyarrow`, `sentence-transformers`), use this type-safe pattern:

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Import for type checking - ensures proper types during static analysis
    from some_optional_package import SomeClass
else:
    # Runtime import handling - gracefully handle missing dependencies
    try:
        from some_optional_package import SomeClass
        OPTIONAL_AVAILABLE = True
    except ImportError:
        SomeClass = None  # type: ignore
        OPTIONAL_AVAILABLE = False

# In your code, check availability before use
if not OPTIONAL_AVAILABLE:
    raise ImportError(
        "Optional package 'some_optional_package' is required for this feature. "
        "Install it with: uv pip install -e '.[feature_name]'"
    )
```

This pattern:
- Prevents type errors during static analysis
- Handles missing dependencies gracefully at runtime
- Provides clear installation instructions to users

## Commands

### Python Environment Setup

```bash
# Create and activate a virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install package in development mode with dev dependencies
uv pip install -e ".[dev]"
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
uv pip install -e ".[docs]"

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

1. **Summarisation Model** (`kura/summarisation.py`): Takes user conversations and summarizes them into task descriptions, with optional disk caching using `diskcache`
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
- `BaseSummaryModel` / `SummaryModel` (`kura/summarisation.py`): Summarize conversations with optional disk caching
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
