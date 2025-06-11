# Summarization

Kura transforms conversations into structured, privacy-preserving summaries using the CLIO framework. It captures user intent, task details, and interaction quality while providing automatic extensibility for custom analysis fields. This enables downstream analysis like clustering and visualization.

The summarization process extracts multiple "facets" from each conversation - specific attributes like the high-level topic, number of turns, languages used, and more. Some facets are computed directly (e.g., conversation length), while others are extracted using AI models (e.g., conversation topic). The summaries are designed to preserve user privacy by avoiding specific identifying details while still capturing the key patterns and trends across conversations.

These structured summaries power Kura's ability to identify broad usage patterns and potential risks, without requiring analysts to know exactly what to look for in advance. The hierarchical organization of summaries allows exploring patterns at different levels of granularity - from high-level categories down to specific conversation types.

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


asyncio.run(main())
```

This extracts structured information including the conversation summary, user request, programming languages mentioned, task description, and quality metrics.

---

## How It Works

Kura uses the CLIO framework with LLMs and structured output via Instructor to extract consistent information from conversations. This provides flexibility in model choice while maintaining consistent structured outputs and automatic extensibility.

Key features:

- **Input:** `Conversation` objects with messages and metadata
- **Output:** `ConversationSummary` objects with structured fields
- **Privacy:** Removes PII and proper nouns automatically (based on CLIO framework)
- **Extensibility:** Automatic field mapping from extended schemas to metadata
- **Concurrency:** Processes multiple conversations in parallel for efficiency
- **Checkpointing:** Caches results to avoid recomputation

---

## Summarization Prompt and Output

!!! info "CLIO Framework"

    At the heart of Kura's summarization process is the CLIO framework - a carefully engineered prompt system developed by Anthropic for privacy-preserving conversation analysis. The prompt is designed to capture key aspects while maintaining privacy and clarity, instructing the model to avoid personally identifiable information (PII) and proper nouns while preserving important context.

    **Reference:** [Clio: Privacy-Preserving Insights into Real-World AI Use](https://www.anthropic.com/research/clio)

For each conversation, the model extracts multiple fields that provide a comprehensive view of the interaction:

- **Summary**: Clear, concise description in no more than two sentences
- **Request**: User's overall intent starting with "The user's overall request..."
- **Task**: Specific action being performed starting with "The task is to..."
- **Languages**: Both human and programming languages mentioned
- **Concerning Score**: Safety assessment on 1-5 scale
- **User Frustration**: Frustration level on 1-5 scale
- **Assistant Errors**: Specific errors made by the assistant

This structured output enables downstream analysis while maintaining consistency across large datasets. The summary object serves as the foundation for Kura's clustering and visualization capabilities, allowing patterns and insights to emerge from the aggregated data.

## Customising Summarisation

Kura's summarization follows a procedural, configurable design where you control behavior through function parameters rather than hidden class configuration. The system provides automatic extensibility through schema inheritance and prompt modification. You can customize three key aspects:

### 1. Modify the Model

Different models offer varying performance, cost, and capability trade-offs. You might choose Claude for better reasoning, GPT-4 for consistency, or local models for privacy. Model configuration happens at initialization time since it affects API clients and connection pooling.

```python
from kura import summarise_conversations
from kura.summarisation import SummaryModel

# Use a different model with custom settings
model = SummaryModel(
    model="anthropic/claude-3-5-sonnet-20241022",
    max_concurrent_requests=10,  # Lower for rate limits
)

summaries = await summarise_conversations(
    conversations,
    model=model
)
```

### 2. Extend the CLIO Prompt

You can extend the default CLIO prompt to focus on domain-specific aspects while preserving the core privacy-preserving analysis framework. Use the `prompt` parameter if you'd like to modify the default summarisation prompt.

```python
# Extend CLIO prompt for technical analysis
summaries = await summarise_conversations(
        conversations,
        model=summary_model,
        checkpoint_manager=checkpoint_mgr,
        prompt="""
        Summarise if the conversation is about alpacas and their management.
        If it is, return "Alpacas" in the summary.
        If it is not, return "Not alpacas but about x" in the summary where x here is the topic of the conversation.

        Here is the conversation:
        <messages>
        {% for message in conversation.messages %}
        <message>{{message.role}}: {{message.content}}</message>
        {% endfor %}
        </messages>
        """,
)
```

You can also extend from the default prompt as seen below

```python
from kura.summarisation import DEFAULT_SUMMARY_PROMPT

summaries = await summarise_conversations(
    conversations,
    model=summary_model,
    checkpoint_manager=checkpoint_mgr,
    prompt=DEFAULT_SUMMARY_PROMPT + "Make it long and verbose please!",
)
```

### 3. Extend with Custom Fields (Automatic Schema Extension)

Kura provides automatic extensibility through schema inheritance. Simply extend `GeneratedSummary` with custom fields, and they'll automatically be included in the `ConversationSummary.metadata` without any additional mapping code.

```python
from pydantic import BaseModel, Field
from kura.types.summarisation import GeneratedSummary

class TechnicalSummary(GeneratedSummary):
    """Extended schema with automatic metadata mapping."""
    frameworks_mentioned: list[str] = Field(description="Programming frameworks discussed")
    complexity_level: str = Field(description="Technical complexity: beginner/intermediate/advanced")
    code_quality_issues: list[str] = Field(description="Code quality problems identified")
    technical_depth: int = Field(description="Technical depth rating 1-10")

summaries = await summarise_conversations(
    conversations,
    model=model,
    response_schema=TechnicalSummary,
)

# Access core CLIO fields directly
print(summaries[0].summary)
print(summaries[0].concerning_score)

# Access custom fields automatically in metadata
print(summaries[0].metadata["frameworks_mentioned"])
print(summaries[0].metadata["technical_depth"])
```

**Key Benefits:**

- **Zero Boilerplate**: Custom fields automatically appear in metadata
- **Type Safety**: Full Pydantic validation for custom fields
- **Backward Compatibility**: Core CLIO fields always available
- **Extensible**: Add any number of custom analysis dimensions

## Complete Example: Custom Technical Analysis

Here's a complete example combining all customization features for technical conversation analysis:

```python
from kura import summarise_conversations
from kura.summarisation import SummaryModel
from kura.types.summarisation import GeneratedSummary
from pydantic import Field

# 1. Define custom schema with automatic metadata mapping
class TechnicalSummary(GeneratedSummary):
    """Extended CLIO analysis for technical conversations."""
    frameworks_mentioned: list[str] = Field(description="Programming frameworks/libraries discussed")
    technical_depth: int = Field(description="Technical complexity rating 1-10")
    solution_effectiveness: str = Field(description="Was the solution effective: yes/no/partial")
    code_snippets_present: bool = Field(description="Were code examples provided?")

# 2. Initialize model with custom configuration
model = SummaryModel(
    model="anthropic/claude-3-5-sonnet-20241022",
    max_concurrent_requests=10
)

# 3. Run analysis with extended prompt and custom schema
summaries = await summarise_conversations(
    conversations,
    model=model,
    response_schema=TechnicalSummary,
    prompt="""
    Additionally analyze:
    - Rate technical depth 1-10 (1=basic concepts, 10=advanced architecture)
    - List specific frameworks/libraries mentioned
    - Assess if the provided solution was effective
    - Note if code examples were included
    """,
)

# 4. Access both CLIO fields and custom technical analysis
for summary in summaries:
    # Core CLIO fields
    print(f"Summary: {summary.summary}")
    print(f"User Frustration: {summary.user_frustration}/5")
    print(f"Languages: {summary.languages}")

    # Custom fields automatically in metadata
    print(f"Technical Depth: {summary.metadata['technical_depth']}/10")
    print(f"Frameworks: {summary.metadata['frameworks_mentioned']}")
    print(f"Solution Effective: {summary.metadata['solution_effectiveness']}")
```

The results can be cached with checkpoint managers and visualized in Kura's interactive UI, providing comprehensive technical conversation insights built on the solid CLIO framework.
