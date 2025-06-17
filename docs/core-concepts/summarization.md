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
- **Disk Caching:** Optional persistent caching for individual summaries

---

## Disk Caching

Kura's `SummaryModel` supports optional disk caching using `diskcache` to store individual conversation summaries persistently. This feature significantly reduces API costs and processing time when working with the same conversations repeatedly.

### Benefits

- **Cost Reduction**: Avoid re-summarizing identical conversations across runs
- **Performance**: Skip expensive LLM calls for cached conversations  
- **Persistence**: Cache survives between program restarts
- **Intelligent Keys**: Cache considers all factors that affect output (model, temperature, prompt, etc.)

### Enabling Caching

Caching is **disabled by default**. Enable it by providing a `cache_dir` parameter:

```python
from kura.summarisation import SummaryModel

# Enable caching with custom directory
model = SummaryModel(
    model="openai/gpt-4o-mini",
    cache_dir="./summary_cache"  # Creates cache directory if needed
)

# Caching disabled (default behavior)
model_no_cache = SummaryModel(
    model="openai/gpt-4o-mini"
    # cache_dir=None (default)
)
```

### How Cache Keys Work

The caching system generates unique keys based on all factors that affect summarization output:

- **Conversation content**: All message roles and content
- **Model configuration**: The specific model being used  
- **Generation parameters**: Temperature, prompt text
- **Schema**: The response schema class name
- **Additional kwargs**: Any other parameters passed to the summarization

This ensures cached results are only used when the exact same summarization would be generated.

```python
# These will use different cache entries:
summaries1 = await model.summarise(conversations, temperature=0.2)
summaries2 = await model.summarise(conversations, temperature=0.5)  # Different temperature

# These will also use different cache entries:
summaries3 = await model.summarise(conversations, response_schema=GeneratedSummary)
summaries4 = await model.summarise(conversations, response_schema=CustomSummary)  # Different schema
```

### Usage Examples

**Basic caching setup:**

```python
from kura import summarise_conversations
from kura.summarisation import SummaryModel

async def main():
    # Initialize model with caching enabled
    model = SummaryModel(
        model="openai/gpt-4o-mini",
        cache_dir="./my_summary_cache"
    )
    
    # First run: generates summaries and caches them
    summaries = await summarise_conversations(conversations, model=model)
    print(f"Generated {len(summaries)} summaries")
    
    # Second run: loads from cache (much faster!)
    summaries_cached = await summarise_conversations(conversations, model=model)
    print(f"Loaded {len(summaries_cached)} summaries from cache")
```

**Working with different cache directories:**

```python
# Separate caches for different model configurations
gpt4_model = SummaryModel(
    model="openai/gpt-4o", 
    cache_dir="./cache/gpt4"
)

claude_model = SummaryModel(
    model="anthropic/claude-3-5-sonnet-20241022",
    cache_dir="./cache/claude"
)

# Each model maintains its own cache
gpt4_summaries = await summarise_conversations(conversations, model=gpt4_model)
claude_summaries = await summarise_conversations(conversations, model=claude_model)
```

**Cache behavior with custom prompts:**

```python
from kura.cache import DiskCacheStrategy

model = SummaryModel(cache=DiskCacheStrategy("./cache"))

# Different prompts create separate cache entries
summaries_default = await model.summarise(
    conversations
    # Uses default CLIO prompt
)

summaries_custom = await model.summarise(
    conversations,
    prompt="Focus specifically on technical complexity and user satisfaction."
)

# Each prompt variation gets its own cached results
```

### Cache Management

The cache is managed automatically, but you can control the cache directory location:

```python
import os
from kura.summarisation import SummaryModel
from kura.cache import DiskCacheStrategy

# Use environment variable for cache location
cache_dir = os.getenv("KURA_CACHE_DIR", "./default_cache")
model = SummaryModel(cache=DiskCacheStrategy(cache_dir))

# Or use project-specific cache directories
model = SummaryModel(cache=DiskCacheStrategy(f"./cache/{project_name}/summaries"))
```

**Cache size considerations:** The cache grows with unique conversations. Each cached summary includes the full `ConversationSummary` object. Monitor cache directory size for large-scale deployments.

**Cache invalidation:** The cache automatically handles invalidation through intelligent key generation. No manual cache clearing is needed unless you want to force regeneration of all summaries.

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
