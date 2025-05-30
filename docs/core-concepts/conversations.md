# Conversations

Conversations are the fundamental data units in Kura's analysis pipeline. This document explains how conversations are structured, loaded, and processed.

## Conversation Structure

In Kura, a conversation is represented by the `Conversation` class from `kura.types.conversation`:

```python
from kura.types import Conversation, Message
from datetime import datetime
from uuid import uuid4

# Create a simple conversation
conversation = Conversation(
    id=str(uuid4()),
    created_at=datetime.now(),
    messages=[
        Message(
            role="user",
            content="Hello, can you help me with a Python question?",
            created_at=str(datetime.now())
        ),
        Message(
            role="assistant",
            content="Of course! What's your Python question?",
            created_at=str(datetime.now())
        ),
        Message(
            role="user",
            content="How do I read a file in Python?",
            created_at=str(datetime.now())
        ),
        Message(
            role="assistant",
            content="To read a file in Python, you can use the built-in open() function...",
            created_at=str(datetime.now())
        )
    ],
    metadata={"source": "example", "category": "programming"}
)
```

### Key Components

Each conversation contains:

- **ID**: A unique identifier for the conversation
- **Created At**: Timestamp for when the conversation was created
- **Messages**: A list of message objects, each with:
  - **Role**: Either "user" or "assistant"
  - **Content**: The text content of the message
  - **Created At**: Timestamp for when the message was sent
- **Metadata**: Optional dictionary of additional information

## Loading Conversations

Kura provides several methods for loading conversations from different sources:

### From Claude Conversation Exports

```python
from kura.types import Conversation

# Load from Claude export
conversations = Conversation.from_claude_conversation_dump("conversations.json")
```

### From Hugging Face Datasets

```python
from kura.types import Conversation

# Load from a Hugging Face dataset
conversations = Conversation.from_hf_dataset(
    "ivanleomk/synthetic-gemini-conversations",
    split="train"
)
```

### Creating Custom Loaders

You can create custom loaders for other data sources by implementing functions that convert your data to `Conversation` objects:

```python
def load_from_custom_format(file_path):
    # Load and parse your custom data format
    data = your_parsing_function(file_path)

    # Convert to Conversation objects
    conversations = []
    for entry in data:
        messages = [
            Message(
                role=msg["speaker"],
                content=msg["text"],
                created_at=msg["timestamp"]
            )
            for msg in entry["messages"]
        ]

        conversation = Conversation(
            id=entry["id"],
            created_at=entry["date"],
            messages=messages,
            metadata=entry.get("meta", {})
        )

        conversations.append(conversation)

    return conversations
```

## Conversation Processing

In the Kura pipeline, conversations go through several processing steps:

1. **Loading**: Conversations are loaded from a source
2. **Summarization**: Each conversation is summarized to capture its core intent
3. **Metadata Extraction**: Optional metadata is extracted from the conversation content
4. **Embedding**: Summaries are converted to vector embeddings
5. **Clustering**: Similar conversations are grouped together

## Working with Message Content

The content of messages can be in various formats, but should generally be text. HTML, Markdown, or other structured formats will be processed as-is, which may affect summarization quality.

When working with message content:

- Clean up any special formatting if needed
- Remove system messages if they don't contribute to the conversation topic
- Ensure message ordering is correct for proper context

## Handling Metadata

Conversations can include metadata, which provides additional context:

```python
# Add metadata when creating conversations
conversations = Conversation.from_hf_dataset(
    "allenai/WildChat-nontoxic",
    metadata_fn=lambda x: {
        "model": x["model"],
        "toxic": x["toxic"],
        "redacted": x["redacted"],
    }
)
```

This metadata can later be used to:
- Filter conversations
- Analyze patterns across different conversation attributes
- Provide additional context for visualization

## Next Steps

Now that you understand how conversations are structured in Kura, you can:

- Learn about the [summarization process](summarization.md)
- See how to load different data formats in the [Procedural API quickstart](../getting-started/tutorial-procedural-api.md)
- Explore configuration options in the [Configuration Guide](../getting-started/configuration.md)
