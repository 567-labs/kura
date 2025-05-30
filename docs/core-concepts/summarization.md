# Summarization

Kura transforms conversations into structured, privacy-preserving summaries that capture user intent, task details, and interaction quality. This enables downstream analysis like clustering and visualization.

---

## Quick Start

Here's the simplest way to summarize conversations:

```python
from kura import summarise_conversations
from kura.summarisation import SummaryModel
from kura.types import Conversation, Message
from datetime import datetime
import asyncio

# Create a conversation
conversation = Conversation(
    chat_id="example-1",
    created_at=datetime.now(),
    messages=[
        Message(
            role="user",
            content="How do I use pandas to read a CSV file?",
            created_at=datetime.now(),
        ),
        Message(
            role="assistant",
            content="You can use pd.read_csv('filename.csv') to read a CSV file.",
            created_at=datetime.now(),
        ),
    ],
    metadata={"query_id": "123"},
)


async def main():
    # Initialize the model
    model = SummaryModel()

    # Summarize conversations
    summaries = await summarise_conversations([conversation], model=model)

    # Each summary contains: summary, request, task, languages, etc.
    print(summaries[0].model_dump_json())
    # > {
    #   "summary": "The user is seeking guidance on how to use a specific library to read a file format commonly used for data storage.",
    #   "request": "The user's overall request for the assistant is to explain how to use a library to read a CSV file.",
    #   "topic": null,
    #   "concerning_score": 1,
    #   "user_frustration": 1,
    #   "languages": [
    #     "english",
    #     "python"
    #   ],
    #   "task": "The task is to provide instructions on reading a CSV file using a specific library.",
    #   "assistant_errors": '',
    #   "chat_id": "example-1",
    #   "metadata": {
    #     "conversation_turns": 2,
    #     "query_id": "123"
    #   },
    #   "embedding": null
    # }


asyncio.run(main()
```

This extracts structured information including the conversation summary, user request, programming languages mentioned, task description, and quality metrics.

---

## How It Works

Kura uses LLMs with structured output via Instructor to extract consistent information from conversations. This provides flexibility in model choice while maintaining consistent structured outputs.

Key features:

- **Input:** `Conversation` objects with messages and metadata
- **Output:** `ConversationSummary` objects with structured fields
- **Privacy:** Removes PII and proper nouns automatically
- **Concurrency:** Processes multiple conversations in parallel for efficiency
- **Checkpointing:** Caches results to avoid recomputation
- **Hooks/Extractors:** Optional extractors can add custom metadata to each summary

---

## Summarization Prompt and Output

At the heart of Kura's summarization process is a carefully engineered prompt that extracts structured information from conversations. The prompt is designed to capture key aspects while maintaining privacy and clarity. It instructs the model to avoid personally identifiable information (PII) and proper nouns, while still preserving important context about the interaction.

For each conversation, the model extracts multiple fields that provide a comprehensive view of the interaction. The core summary field provides a clear, concise description in no more than two sentences. The request field captures the user's overall intent, while the task field describes the specific action being performed.

To track the quality and nature of interactions, the model also assesses user frustration and safety concerns on a 1-5 scale, and identifies any errors made by the assistant.

This structured output enables downstream analysis while maintaining consistency across large datasets. The summary object serves as the foundation for Kura's clustering and visualization capabilities, allowing patterns and insights to emerge from the aggregated data.

## Customising Summarisation

Kura's summarization follows a procedural, configurable design where you control behavior through function parameters rather than hidden class configuration. You can customize four key aspects:

### 1. Modify the Model

Different models offer varying performance, cost, and capability trade-offs. You might choose Claude for better reasoning, GPT-4 for consistency, or local models for privacy. Model configuration happens at initialization time since it affects API clients and connection pooling.

```python
from kura import summarise_conversations
from kura.summarisation import SummaryModel

# Use a different model with custom settings
model = SummaryModel(
    model="anthropic/claude-3-5-sonnet-20241022",
    max_concurrent_requests=10,  # Lower for rate limits
    checkpoint_filename="claude_summaries.jsonl"
)

summaries = await summarise_conversations(
    conversations,
    model=model
)
```

### 2. Modify the Prompt

Custom prompts let you focus extraction on domain-specific aspects like technical complexity, emotional tone, or business requirements. This is especially useful when analyzing specialized conversations where the default prompt misses important context.

For each summary you generate, we provide each [Conversation](../api/index.md#core-classes) object which you pass into the main `summarise_conversations` function. This makes it easy to write and customise your prompts to focus on the specific issues or domain that you're dealing with.

This Conversation object has the following properties

`messages`: A list of message objects, each containing:

- `role`: Either "user" or "assistant"
- `content`: The message text
- `created_at`: Timestamp of when the message was created

`metadata`: A dictionary containing any additional metadata associated with the conversation

```python
TECHNICAL_PROMPT = """
Analyze this technical support conversation, focusing on:
- Problem complexity and resolution approach
- User's technical expertise level
- Whether the solution was effective

<messages>
{% for message in messages %}
<message>{{message.role}}: {{message.content}}</message>
{% endfor %}
</messages>

Extract structured information about the technical interaction...
"""

summaries = await summarise_conversations(
    conversations,
    model=model,
    prompt_template=TECHNICAL_PROMPT
)
```

### 3. Modify the Structured Output

Custom response schemas capture domain-specific metadata that the default ConversationSummary doesn't include. This enables specialized analysis workflows like technical debt assessment, customer satisfaction tracking, or compliance monitoring.

```python
from pydantic import BaseModel, Field
from kura.types.summarisation import GeneratedSummary

class TechnicalSummary(GeneratedSummary):
    frameworks_mentioned: list[str] = Field(description="Programming frameworks discussed")
    complexity_level: str = Field(description="Technical complexity: beginner/intermediate/advanced")
    code_quality_issues: list[str] = Field(description="Code quality problems identified")

summaries = await summarise_conversations(
    conversations,
    model=model,
    response_schema=TechnicalSummary
)
```

### 4. Add Custom Summary Converters

Summary converters transform the raw LLM output into your final data structure, adding computed fields or reformatting data. This is where you merge conversation metadata with extracted content and apply business logic.

This is then cached in the directory you specify if you're defined a checkpoint manager, allowing you to store and leverage this data in our interactive visualiser UI.

```python
from kura.types import ConversationSummary

def technical_summary_converter(
    summary: TechnicalSummary,
    conversation: Conversation
) -> ConversationSummary:
    """Convert technical summary with additional computed fields."""
    return ConversationSummary(
        chat_id=conversation.chat_id,
        metadata={
            "conversation_turns": len(conversation.messages),
            "has_code": len(summary.frameworks_mentioned) > 0,
            "complexity_score": {"beginner": 1, "intermediate": 2, "advanced": 3}.get(
                summary.complexity_level, 0
            ),
            **conversation.metadata,
        },
        **summary.model_dump(),
    )

summaries = await summarise_conversations(
    conversations,
    model=model,
    response_schema=TechnicalSummary,
    summary_converter=technical_summary_converter
)
```

All these parameters can be combined for complete customization while maintaining Kura's procedural, composable design. The key insight is that configuration happens through explicit function parameters, not hidden class state.
