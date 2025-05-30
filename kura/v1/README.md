# Kura V1: Procedural Implementation

This directory contains a procedural, functional approach to the Kura conversation analysis pipeline. Instead of using a single orchestrating class, this implementation breaks the pipeline down into composable functions that can be used independently or together.

## Design Philosophy

### **Function-Based Architecture**
The core design principle is **functions orchestrate, models execute**. Each pipeline step is a pure function that:
- Takes explicit inputs (data + model + optional checkpoint manager)
- Returns explicit outputs (processed data)
- Has no hidden state or side effects
- Works with any model that implements the required interface

### **Polymorphism Through Interfaces**
All functions work with **heterogeneous models** through base class interfaces:
- `BaseSummaryModel` - Instructor, vLLM, Hugging Face, local models, etc.
- `BaseClusterModel` - HDBSCAN, KMeans, custom clustering algorithms
- `BaseMetaClusterModel` - Different hierarchical clustering strategies
- `BaseDimensionalityReduction` - UMAP, t-SNE, PCA, etc.

### **Keyword-Only Arguments**
All functions use keyword-only arguments for:
- **Explicit API** - Clear what each parameter does
- **Maintainability** - Easy to add new parameters without breaking changes
- **Readability** - Self-documenting function calls

## Core API

### Pipeline Functions
- `summarise_conversations(conversations, model, checkpoint_manager=None)` - Generate summaries from conversations
- `generate_base_clusters_from_conversation_summaries(summaries, model, checkpoint_manager=None)` - Create initial clusters from summaries  
- `reduce_clusters_from_base_clusters(clusters, model, checkpoint_manager=None)` - Build hierarchical cluster structure
- `reduce_dimensionality_from_clusters(clusters, model, checkpoint_manager=None)` - Project clusters to 2D for visualization

### Utilities
- `CheckpointManager(checkpoint_dir, enabled=True)` - Handles checkpoint loading/saving

## Key Benefits

### **Better Composability**
- Use individual functions for maximum flexibility
- Mix and match different steps as needed
- Easy to experiment with different models or configurations

### **Easier Testing**
- Test individual pipeline steps in isolation
- Mock dependencies more easily
- Clearer dependency injection

### **Clearer Data Flow**
- Function signatures make inputs/outputs explicit
- No hidden state or side effects
- Better support for functional programming patterns

### **More Flexible**
- Skip steps you don't need
- Run steps in different orders
- Easier to parallelize or optimize individual components

### **Heterogeneous Model Support**
- Same function interface works with any model implementation
- Easy A/B testing between different model types
- Configuration flexibility for different deployment scenarios

## Usage Examples

### 1. Basic Pipeline

```python
import asyncio
import logging
from kura import (
    summarise_conversations, 
    generate_base_clusters_from_conversation_summaries, 
    reduce_clusters_from_base_clusters,
    reduce_dimensionality_from_clusters,
    CheckpointManager
)
from kura.summarisation import SummaryModel
from kura.cluster import ClusterModel
from kura.meta_cluster import MetaClusterModel
from kura.dimensionality import HDBUMAP

# Set up logging
logging.basicConfig(level=logging.INFO)

async def analyze_conversations(conversations):
    # Set up models
    summary_model = SummaryModel()
    cluster_model = ClusterModel()
    meta_cluster_model = MetaClusterModel()
    dimensionality_model = HDBUMAP()
    
    # Set up checkpointing (optional)
    checkpoint_manager = CheckpointManager("./checkpoints", enabled=True)
    
    # Run pipeline steps
    summaries = await summarise_conversations(
        conversations, 
        summary_model, 
        checkpoint_manager
    )
    
    clusters = await generate_base_clusters_from_conversation_summaries(
        summaries, 
        cluster_model, 
        checkpoint_manager
    )
    
    reduced_clusters = await reduce_clusters_from_base_clusters(
        clusters, 
        meta_cluster_model, 
        checkpoint_manager
    )
    
    projected = await reduce_dimensionality_from_clusters(
        reduced_clusters, 
        dimensionality_model, 
        checkpoint_manager
    )
    
    return projected

# Run it
conversations = load_your_conversations()
result = asyncio.run(analyze_conversations(conversations))
```

### 2. Heterogeneous Models

The function-based approach shines when working with different model types:

```python
# Define different summary model implementations
class OpenAISummaryModel(BaseSummaryModel):
    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
    
    async def summarise(self, conversations):
        # Async OpenAI API calls
        tasks = [self._summarise_one(conv) for conv in conversations]
        return await asyncio.gather(*tasks)

class VLLMSummaryModel(BaseSummaryModel):
    def __init__(self, model_path: str, host: str = "localhost", port: int = 8000):
        self.model_path = model_path
        self.base_url = f"http://{host}:{port}"
    
    async def summarise(self, conversations):
        # Async calls to local vLLM server
        async with aiohttp.ClientSession() as session:
            tasks = [self._call_vllm(session, conv) for conv in conversations]
            return await asyncio.gather(*tasks)

class HuggingFaceSummaryModel(BaseSummaryModel):
    def __init__(self, model_name: str):
        from transformers import pipeline
        self.summarizer = pipeline("summarization", model=model_name)
    
    async def summarise(self, conversations):
        # Run sync operations in thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            self._summarise_batch, 
            conversations
        )

# Usage - Same function, different models!
async def compare_models(conversations):
    models = [
        OpenAISummaryModel(api_key="sk-...", model="gpt-4"),
        VLLMSummaryModel("/path/to/llama", host="gpu-server"),
        HuggingFaceSummaryModel("facebook/bart-large-cnn")
    ]
    
    results = []
    for model in models:
        summaries = await summarise_conversations(
            conversations,
            model=model,
            checkpoint_manager=None  # Disable checkpointing for comparison
        )
        results.append((type(model).__name__, summaries))
    
    return results
```

### 3. Custom Pipeline (Skip Steps)

```python
async def custom_pipeline(conversations):
    summary_model = SummaryModel()
    cluster_model = ClusterModel()
    dimensionality_model = HDBUMAP()
    
    # Generate summaries
    summaries = await summarise_conversations(
        conversations,
        model=summary_model,
        checkpoint_manager=None
    )
    
    # Generate base clusters
    clusters = await generate_base_clusters_from_conversation_summaries(
        summaries,
        model=cluster_model,
        checkpoint_manager=None
    )
    
    # Skip meta-clustering, go straight to dimensionality reduction
    projected = await reduce_dimensionality_from_clusters(
        clusters,
        model=dimensionality_model,
        checkpoint_manager=None
    )
    
    return projected
```

### 4. A/B Testing Different Models

```python
async def ab_test_clustering(summaries):
    """Test different clustering algorithms."""
    
    # Test different clustering models
    hdbscan_model = ClusterModel(algorithm="hdbscan", min_cluster_size=5)
    kmeans_model = ClusterModel(algorithm="kmeans", n_clusters=10)
    
    # Use same checkpoint manager for fair comparison
    checkpoint_mgr = CheckpointManager("./ab_test", enabled=False)
    
    # Run both models
    hdbscan_clusters = await generate_base_clusters_from_conversation_summaries(
        summaries,
        model=hdbscan_model,
        checkpoint_manager=checkpoint_mgr
    )
    
    kmeans_clusters = await generate_base_clusters_from_conversation_summaries(
        summaries,
        model=kmeans_model, 
        checkpoint_manager=checkpoint_mgr
    )
    
    return {
        "hdbscan": hdbscan_clusters,
        "kmeans": kmeans_clusters
    }
```

### 5. Parallel Processing

```python
import asyncio

async def parallel_analysis(conversation_batches):
    """Process multiple conversation batches in parallel."""
    summary_model = SummaryModel()
    
    # Process summaries in parallel
    summary_tasks = [
        summarise_conversations(
            batch,
            model=summary_model,
            checkpoint_manager=None
        )
        for batch in conversation_batches
    ]
    
    all_summaries = await asyncio.gather(*summary_tasks)
    
    # Flatten and continue with rest of pipeline
    flattened_summaries = [s for batch in all_summaries for s in batch]
    
    # Continue with clustering...
    cluster_model = ClusterModel()
    clusters = await generate_base_clusters_from_conversation_summaries(
        flattened_summaries,
        model=cluster_model,
        checkpoint_manager=None
    )
    
    return clusters
```

### 6. Configuration-Based Model Selection

```python
def create_summary_model(config):
    """Factory function for creating different model types."""
    if config["type"] == "openai":
        return OpenAISummaryModel(
            api_key=config["api_key"],
            model=config.get("model", "gpt-4")
        )
    elif config["type"] == "vllm":
        return VLLMSummaryModel(
            model_path=config["model_path"],
            host=config.get("host", "localhost"),
            port=config.get("port", 8000)
        )
    elif config["type"] == "huggingface":
        return HuggingFaceSummaryModel(config["model_name"])
    else:
        raise ValueError(f"Unknown model type: {config['type']}")

async def configurable_pipeline(conversations, model_config):
    """Run pipeline with configurable model selection."""
    
    # Create model based on configuration
    summary_model = create_summary_model(model_config)
    
    # Run pipeline
    checkpoint_mgr = CheckpointManager("./production", enabled=True)
    summaries = await summarise_conversations(
        conversations,
        model=summary_model,
        checkpoint_manager=checkpoint_mgr
    )
    
    return summaries

# Usage
openai_config = {"type": "openai", "api_key": "sk-...", "model": "gpt-4"}
vllm_config = {"type": "vllm", "model_path": "/models/llama", "host": "gpu-server"}

summaries1 = await configurable_pipeline(conversations, openai_config)
summaries2 = await configurable_pipeline(conversations, vllm_config)
```

## Checkpoint Management

```python
# Enable checkpointing
checkpoint_manager = CheckpointManager("./my_analysis", enabled=True)

# Disable checkpointing
no_checkpoints = CheckpointManager("./temp", enabled=False)

# Or pass None to any function to disable checkpointing
summaries = await summarise_conversations(
    conversations,
    model=summary_model,
    checkpoint_manager=None
)
```

## Logging

The procedural implementation uses Python's standard logging module:

```python
import logging

# Basic logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Or get the specific logger
logger = logging.getLogger('kura.kura')
logger.setLevel(logging.DEBUG)
```



## Design Advantages

### 1. **Polymorphism Through Interfaces**
- Functions work with any model that implements the base interface
- Easy to swap different implementations (OpenAI ↔ vLLM ↔ local models)
- Models encapsulate their complexity (async vs sync, API vs local, etc.)

### 2. **Function-Based Orchestration**
- **Functions orchestrate, models execute**
- No mixed concerns - models focus on their core logic
- Checkpoint management is separate and optional
- Better separation of concerns

### 3. **Explicit Keyword Arguments**
- Self-documenting function calls
- Easy to add new parameters without breaking changes
- Reduces parameter ordering mistakes
- Improves code readability

### 4. **Composability and Flexibility**
- Mix and match pipeline steps as needed
- Test individual functions in isolation with clear inputs/outputs
- Debug specific steps, inspect intermediate results
- A/B test different models easily
- Optimize or parallelize individual steps
- Support for functional programming patterns

### 5. **Better Testing**
```python
# Easy to test with mocks
def test_summarise_conversations():
    mock_model = Mock()
    mock_checkpoint = Mock()
    
    await summarise_conversations(
        conversations=test_conversations,
        model=mock_model,
        checkpoint_manager=mock_checkpoint
    )
    
    mock_model.summarise.assert_called_once_with(test_conversations)
```

### 6. **Logging Instead of Print Statements**
- Proper Python logging with configurable levels
- Better for production deployments
- Easy to integrate with existing logging infrastructure

This procedural implementation provides a **clean, flexible, and maintainable** approach to conversation analysis that scales from research experimentation to production deployments. 