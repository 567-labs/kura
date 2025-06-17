# Cached Embeddings

The `CachedEmbeddingModel` provides automatic caching for embedding models to avoid recomputing embeddings for the same inputs. This can significantly reduce API costs and improve performance when working with large datasets.

## Features

- **Automatic caching**: Embeddings are automatically cached based on model name, text content, and additional parameters
- **Cache key generation**: Uses SHA256 hashing of model parameters and text content for consistent cache keys
- **Partial cache hits**: Efficiently handles scenarios where some texts are cached and others need embedding
- **Configurable**: Supports any `BaseEmbeddingModel` and `BaseCheckpointManager` implementation
- **Type safe**: Full type hints and error handling

## Basic Usage

```python
import asyncio
from kura import CachedEmbeddingModel
from kura.embedding import OpenAIEmbeddingModel
from kura.checkpoints import JSONLCheckpointManager

async def main():
    # Create base embedding model
    base_model = OpenAIEmbeddingModel(model_name="text-embedding-3-small")
    
    # Create cache manager
    cache_manager = JSONLCheckpointManager("./embeddings_cache")
    
    # Create cached embedding model
    cached_model = CachedEmbeddingModel(
        embedding_model=base_model,
        cache_manager=cache_manager
    )
    
    # Use like any other embedding model
    texts = ["Hello world", "Test sentence"]
    embeddings = await cached_model.embed(texts)
    
    # Second call will use cache
    cached_embeddings = await cached_model.embed(texts)
    assert embeddings == cached_embeddings

asyncio.run(main())
```

## Advanced Configuration

### Custom Cache Parameters

You can include additional parameters in the cache key to ensure different configurations don't share cached embeddings:

```python
cached_model = CachedEmbeddingModel(
    embedding_model=base_model,
    cache_manager=cache_manager,
    # These parameters will be included in cache keys
    temperature=0.5,
    custom_preprocessing="lowercase",
    batch_size=50
)
```

### Different Cache Backends

Use any checkpoint manager implementation:

```python
# JSONL caching
from kura.checkpoints import JSONLCheckpointManager
cache_manager = JSONLCheckpointManager("./cache")

# Parquet caching (better compression)
from kura.checkpoints import ParquetCheckpointManager
cache_manager = ParquetCheckpointManager("./cache")

# HuggingFace datasets caching
from kura.checkpoints import HFDatasetCheckpointManager
cache_manager = HFDatasetCheckpointManager("./cache")
```

### Disabling Cache

```python
# Without cache manager - behaves like original model
cached_model = CachedEmbeddingModel(base_model, cache_manager=None)

# With disabled cache manager
cache_manager = JSONLCheckpointManager("./cache", enabled=False)
cached_model = CachedEmbeddingModel(base_model, cache_manager)
```

## Cache Key Generation

Cache keys are generated using SHA256 hashing of:

1. **Model name**: `embedding_model.model_name`
2. **Text content**: The exact text being embedded
3. **Additional kwargs**: Any extra parameters passed during initialization

This ensures that:
- Same text with same model and parameters = cache hit
- Different text = different cache key
- Same text with different model/parameters = different cache key

## Performance Considerations

### Cache Hit Rate

The cached embedding model logs information about cache performance:

```
INFO:kura.cached_embedding:Found 8 cached embeddings, 2 need embedding
```

### Storage Requirements

Each embedding is stored as a separate file. For large-scale usage, consider:

- Using `ParquetCheckpointManager` for better compression
- Implementing custom cleanup strategies for old cache entries
- Monitoring cache directory size

### Memory Usage

The current implementation loads cache entries individually. For very large caches, consider batching cache lookups.

## Error Handling

### Missing Model Name

```python
class BadModel(BaseEmbeddingModel):
    # Missing model_name attribute
    async def embed(self, texts): return []
    def slug(self): return "bad"

# This will raise ValueError
CachedEmbeddingModel(BadModel(), cache_manager)
```

### Cache Failures

Cache operations are designed to be resilient:
- Cache load failures fall back to fresh embedding
- Cache save failures are logged but don't interrupt processing
- Invalid cache data is ignored

## Integration Examples

### With Conversation Summarization

```python
from kura import summarise_conversations, CachedEmbeddingModel
from kura.embedding import OpenAIEmbeddingModel
from kura.checkpoints import ParquetCheckpointManager

# Create cached embedding model
base_model = OpenAIEmbeddingModel()
cache_manager = ParquetCheckpointManager("./summary_cache")
cached_model = CachedEmbeddingModel(base_model, cache_manager)

# Use in summarization pipeline
summaries = await summarise_conversations(
    conversations=conversations,
    embedding_model=cached_model,  # Will cache embeddings
    # ... other parameters
)
```

### With Custom Embedding Models

```python
from kura.embedding import SentenceTransformerEmbeddingModel

# Works with any BaseEmbeddingModel implementation
st_model = SentenceTransformerEmbeddingModel(
    model_name="all-MiniLM-L6-v2"
)

cached_st_model = CachedEmbeddingModel(
    embedding_model=st_model,
    cache_manager=cache_manager,
    device="cpu"  # Additional parameter for cache key
)
```

## Best Practices

1. **Use meaningful cache parameters**: Include all parameters that affect embedding output
2. **Monitor cache size**: Implement cleanup strategies for production use
3. **Choose appropriate cache backend**: Use Parquet for better compression on large datasets
4. **Test cache behavior**: Verify that cached and fresh embeddings match
5. **Handle cache directory permissions**: Ensure write access to cache directory

## Troubleshooting

### Cache Not Working

1. Check if cache manager is enabled: `cache_manager.is_enabled()`
2. Verify cache directory permissions
3. Check logs for cache save/load errors
4. Ensure model has `model_name` attribute

### Inconsistent Cache Keys

1. Verify all relevant parameters are included in kwargs
2. Check that text preprocessing is consistent
3. Ensure model_name is stable across runs

### Performance Issues

1. Monitor cache hit rate in logs
2. Consider cache cleanup for old entries
3. Use appropriate cache backend for your use case
4. Profile memory usage with large caches
