# Kura Development Guidelines

## Core Philosophy

Kura prioritizes procedural, composable functions over class-based inheritance hierarchies. Functions should be configurable, testable, and extensible through parameters, not buried implementation details.

## Architecture Principles

### Procedural First, Classes Second

Prefer pure functions with explicit parameters over hardcoded class implementations. All important configuration should be exposed as function parameters, not hidden in class constructors.

```python
# Good: Configurable function
async def summarise_conversations(
    conversations: List[Conversation],
    model: str = "openai/gpt-4o-mini",
    temperature: float = 0.2,
    **kwargs
) -> List[ConversationSummary]:
    pass

# Bad: Configuration buried in class
class SummaryModel:
    def __init__(self):
        self.model = "gpt-4o-mini"  # Hardcoded
        self.temperature = 0.2      # Hardcoded
```

### Composition Over Inheritance

Use dependency injection instead of inheritance hierarchies. Pass dependencies as parameters to make functions easily testable and configurable.

```python
# Good: Dependencies injected
async def cluster_conversations(
    conversations: List[Conversation],
    embedding_model: BaseEmbeddingModel,
    summary_model: BaseSummaryModel,
) -> List[Cluster]:
    pass
```

## Code Guidelines

### Function Signatures

Make function signatures explicit and configurable. Expose all important parameters as function arguments rather than hiding them in implementation details.

```python
async def process_documents(
    documents: List[Document],
    embedding_model: BaseEmbeddingModel,
    batch_size: int = 100,
    max_retries: int = 3,
    **kwargs
) -> List[ProcessedDocument]:
    pass
```

### Base Classes

Keep base classes minimal and focused on single responsibilities. Avoid bloated interfaces that mix multiple concerns.

```python
class BaseSummaryModel(ABC):
    @abstractmethod
    async def summarise(
        self,
        conversations: List[Conversation],
        **kwargs
    ) -> List[ConversationSummary]:
        pass
```

### Error Handling

Use specific error types and implement retry logic where appropriate. Make errors recoverable when possible.

```python
class ModelTimeoutError(Exception):
    """Raised when model request times out."""
    pass

async def embed_with_retry(
    texts: List[str],
    model: BaseEmbeddingModel,
    max_retries: int = 3,
) -> List[List[float]]:
    for attempt in range(max_retries):
        try:
            return await model.embed(texts)
        except ModelTimeoutError:
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(2 ** attempt)
```

## Anti-Patterns

Avoid these common mistakes:

- Hidden configuration (LangChain mistake): Don't bury important parameters in class constructors
- Inheritance traps: Don't force inheritance for customization when composition works better
- Monolithic functions: Break down large functions with mixed responsibilities

## Documentation Guidelines

### Writing Style

Write clear, direct explanations without unnecessary fluff or fancy terminology. Focus on practical information that helps users understand and use Kura effectively.

Use paragraphs instead of bullet points where possible. Bullet points should be reserved for genuine lists or step-by-step instructions where the order matters.

Never use emojis in documentation. The professional tone should come from clear, helpful content, not decorative elements.

### Content Structure

Start with core concepts before diving into specific implementation details. Users should understand what something does and why they need it before seeing how to implement it.

Understand the purpose of each page relative to others. If it's a quickstart guide, the goal is to pique interest and show immediate value, then link to comprehensive tutorials. If it's a reference page, focus on completeness and accuracy.

Use descriptive headers that clearly indicate what content follows. Avoid generic headers like "Overview" when something more specific would be clearer.

### Code Examples

Provide complete, runnable examples rather than fragments. Users should be able to copy and execute the code without guessing missing imports or setup.

Show the expected output or result when helpful. Users benefit from seeing what success looks like, especially for complex operations.

Explain the reasoning behind configuration choices instead of just listing parameters.

```python
# Good: Explains the reasoning
summary_model = SummaryModel(
    temperature=0.1,  # Lower temperature for more consistent summaries
    cache_dir=".cache"  # Speeds up re-runs by 85x
)
```

### MkDocs Features

Use MkDocs-specific functionality to improve readability, but sparingly. Aim for 1-2 callouts or cards per page maximum.

Callouts work well for important notes, warnings, or tips that complement the main content. Cards are effective for showing different options or pathways.

Keep each page self-contained with focused, digestible information. Users should be able to understand a concept completely from a single page without jumping between multiple sections.

if you're using a callout/Admonition, make sure to put it under a header in the form

```
#Header
!!!info "Info"
    This is an info callout.

<some quick paragraph here explaining everything about this specific callout>
```

## Command Reference

### Build Commands

```bash
# Type checking
uv run mypy kura/

# Linting
uv run ruff check

# Testing
uv run pytest tests/

# Documentation
uv run mkdocs serve
```

### Development Workflow

1. Always expose configuration as parameters
2. Write tests for individual functions, not just end-to-end
3. Use dependency injection for external services
4. Prefer composition over inheritance
5. Make functions pure when possible (no hidden state)
