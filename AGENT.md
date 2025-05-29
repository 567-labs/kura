# Kura Development Guidelines

## Core Philosophy: Procedural API Design

Kura prioritizes **procedural, composable functions** over class-based inheritance hierarchies. Functions should be configurable, testable, and extensible through parameters, not buried implementation details.

## Architecture Principles

### 1. **Procedural First, Classes Second**

**✅ Preferred: Pure Functions**

```python
# Good: Configurable function with explicit parameters
async def summarise_conversations(
    conversations: List[Conversation],
    model: str = "openai/gpt-4o-mini",
    response_schema: Type[BaseModel] = GeneratedSummary,
    prompt_template: str = DEFAULT_PROMPT,
    extractors: List[ExtractorFunction] = None,
    temperature: float = 0.2,
    **kwargs
) -> List[ConversationSummary]:
    pass
```

**❌ Avoid: Hardcoded Class Implementation**

```python
# Bad: Configuration buried in class, not extensible
class SummaryModel:
    def __init__(self):
        self.model = "gpt-4o-mini"  # Hardcoded
        self.temperature = 0.2      # Hardcoded
        self.prompt = HARDCODED_PROMPT  # Not configurable

    async def summarise(self, conversations):  # No flexibility
        pass
```

### 2. **Configuration as Parameters, Not Implementation Details**

**✅ Expose Everything Configurable**

```python
# Good: All important parameters exposed
class BaseEmbeddingModel(ABC):
    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        pass

class OpenAIEmbeddingModel(BaseEmbeddingModel):
    def __init__(
        self,
        model_name: str = "text-embedding-3-small",  # ✅ Configurable
        batch_size: int = 50,                        # ✅ Configurable
        concurrent_jobs: int = 5,                    # ✅ Configurable
    ):
        pass
```

**❌ Hardcoded Internal Configuration**

```python
# Bad: Important settings buried in implementation
class SummaryModel:
    async def summarise(self, conversations):
        # Bad: Temperature hardcoded
        resp = await client.chat.completions.create(
            temperature=0.2,  # ❌ Should be parameter
            response_model=GeneratedSummary,  # ❌ Should be configurable
        )
```

### 3. **Composition Over Inheritance**

**✅ Dependency Injection**

```python
# Good: Dependencies passed in, easily testable
async def cluster_conversations(
    conversations: List[Conversation],
    embedding_model: BaseEmbeddingModel,
    summary_model: BaseSummaryModel,
    clustering_method: BaseClusteringMethod,
) -> List[Cluster]:
    pass
```

**❌ Inheritance-Heavy Design**

```python
# Bad: Forces inheritance, hard to test
class Kura:
    def __init__(self):
        self.embedding_model = OpenAIEmbeddingModel()  # ❌ Hardcoded
        self.summary_model = SummaryModel()            # ❌ Hardcoded
```

## Specific Code Guidelines

### 1. **Function Signatures: Be Explicit**

**✅ Clear, Configurable Signatures**

```python
async def process_documents(
    documents: List[Document],
    embedding_model: BaseEmbeddingModel,
    batch_size: int = 100,
    max_retries: int = 3,
    timeout_seconds: float = 30.0,
    **kwargs
) -> List[ProcessedDocument]:
    """
    Process documents with configurable parameters.

    Args:
        documents: Input documents to process
        embedding_model: Model for generating embeddings
        batch_size: Number of documents per batch
        max_retries: Maximum retry attempts on failure
        timeout_seconds: Request timeout in seconds
        **kwargs: Additional model-specific parameters
    """
    pass
```

**❌ Vague or Hardcoded Signatures**

```python
# Bad: No configuration options
async def process_documents(documents):
    # All behavior hardcoded inside
    pass
```

### 2. **Base Classes: Minimal, Focused Interfaces**

**✅ Clean Abstract Interfaces**

```python
class BaseSummaryModel(ABC):
    @abstractmethod
    async def summarise(
        self,
        conversations: List[Conversation],
        response_schema: Optional[Type[BaseModel]] = None,
        prompt_template: Optional[str] = None,
        extractors: Optional[List[ExtractorFunction]] = None,
        **kwargs
    ) -> List[ConversationSummary]:
        """Single responsibility: convert conversations to summaries."""
        pass
```

**❌ Bloated Base Classes**

```python
# Bad: Too many responsibilities
class BaseProcessor(ABC):
    @abstractmethod
    def process(self, data): pass

    @abstractmethod
    def validate(self, data): pass

    @abstractmethod
    def save(self, data): pass  # ❌ Should be separate
```

### 3. **Error Handling: Explicit and Recoverable**

**✅ Specific Error Types and Recovery**

```python
class ModelTimeoutError(Exception):
    """Raised when model request times out."""
    pass

async def embed_with_retry(
    texts: List[str],
    model: BaseEmbeddingModel,
    max_retries: int = 3,
    backoff_factor: float = 2.0,
) -> List[List[float]]:
    for attempt in range(max_retries):
        try:
            return await model.embed(texts)
        except ModelTimeoutError as e:
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(backoff_factor ** attempt)
```

### 4. **Testing: Design for Testability**

**✅ Easy to Mock and Test**

```python
# Good: Dependencies injected, easy to test
async def test_summarisation():
    mock_model = Mock(spec=BaseSummaryModel)
    mock_model.summarise.return_value = [mock_summary]

    result = await summarise_conversations(
        conversations=[test_conversation],
        model=mock_model
    )

    assert len(result) == 1
```

## Anti-Patterns to Avoid

### 1. **The LangChain Mistake: Hidden Configuration**

```python
# ❌ Bad: Important behavior hidden
class ChainProcessor:
    def __init__(self):
        self._temperature = 0.7  # Hidden
        self._max_tokens = 1000  # Hidden

    def process(self, input):
        # User can't control temperature or max_tokens
        pass
```

### 2. **The Inheritance Trap**

```python
# ❌ Bad: Forces inheritance for customization
class BaseAnalyzer:
    def analyze(self, data):
        step1 = self.preprocess(data)  # Must override
        step2 = self.transform(step1)  # Must override
        return self.postprocess(step2) # Must override
```

### 3. **The Monolithic Function**

```python
# ❌ Bad: Does too much, hard to test parts
async def analyze_everything(conversations):
    # 200 lines of mixed responsibilities
    summaries = generate_summaries()  # Hardcoded model
    clusters = create_clusters()      # Hardcoded algorithm
    visualizations = create_viz()     # Hardcoded style
    return everything
```

## Recommended Patterns

### 1. **Builder Pattern for Complex Configuration**

```python
class SummaryModelBuilder:
    def __init__(self):
        self.model = "openai/gpt-4o-mini"
        self.temperature = 0.2
        self.extractors = []

    def with_model(self, model: str) -> "SummaryModelBuilder":
        self.model = model
        return self

    def with_temperature(self, temp: float) -> "SummaryModelBuilder":
        self.temperature = temp
        return self

    def with_extractors(self, extractors: List[ExtractorFunction]) -> "SummaryModelBuilder":
        self.extractors = extractors
        return self

    def build(self) -> SummaryModel:
        return SummaryModel(
            model=self.model,
            temperature=self.temperature,
            default_extractors=self.extractors
        )
```

### 2. **Pipeline Pattern for Sequential Processing**

```python
async def conversation_analysis_pipeline(
    conversations: List[Conversation],
    steps: List[ProcessingStep],
    checkpoint_manager: Optional[CheckpointManager] = None,
) -> ProcessingResult:
    """Configurable pipeline with optional checkpointing."""
    result = ProcessingResult(conversations=conversations)

    for step in steps:
        result = await step.process(result, checkpoint_manager)

    return result
```

### 3. **Registry Pattern for Extensibility**

```python
class ModelRegistry:
    _models: Dict[str, Type[BaseModel]] = {}

    @classmethod
    def register(cls, name: str, model_class: Type[BaseModel]):
        cls._models[name] = model_class

    @classmethod
    def create(cls, name: str, **kwargs) -> BaseModel:
        if name not in cls._models:
            raise ValueError(f"Unknown model: {name}")
        return cls._models[name](**kwargs)

# Usage
ModelRegistry.register("openai-summary", OpenAISummaryModel)
model = ModelRegistry.create("openai-summary", temperature=0.1)
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

1. **Always expose configuration as parameters**
2. **Write tests for individual functions, not just end-to-end**
3. **Use dependency injection for external services**
4. **Prefer composition over inheritance**
5. **Make functions pure when possible (no hidden state)**
