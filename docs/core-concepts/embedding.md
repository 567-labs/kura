# Embedding

Kura converts text into numerical vectors (embeddings) for clustering and similarity analysis. Embeddings are automatically generated during clustering, but you can configure which model to use.

---

## Embedding Models

**By default, Kura uses OpenAI embeddings** (`text-embedding-3-small`). You can easily switch to other models if you want to experiment with different embedding providers or run models locally.

Choose between cloud API models and local models:

### Cloud Models (API-based)

Cloud models offer high quality embeddings but require API keys and internet connectivity.

```python
from kura.embedding import OpenAIEmbeddingModel, CohereEmbeddingModel

# OpenAI Embeddings
# - text-embedding-3-small: 1536 dimensions, cost-effective
# - text-embedding-3-large: 3072 dimensions, highest performance
openai_model = OpenAIEmbeddingModel(
    model_name="text-embedding-3-large",  # or "text-embedding-3-small"
    model_batch_size=100,
    n_concurrent_jobs=10
)

# Cohere Embeddings
# - embed-v4.0: Latest model, optimized for various tasks
# - Supports clustering, search, and classification input types
cohere_model = CohereEmbeddingModel(
    model_name="embed-v4.0",
    model_batch_size=96,
    n_concurrent_jobs=5,
    input_type="clustering",  # Options: "clustering", "search_query", "search_document"
    api_key=None  # Uses COHERE_API_KEY environment variable
)
```

### Local Models (Self-hosted)

Local models run on your machine without API calls, ensuring privacy and eliminating API costs.

```python
from kura.embedding import SentenceTransformerEmbeddingModel

# Sentence Transformers - runs locally
# Popular models: all-MiniLM-L6-v2, all-mpnet-base-v2, e5-large-v2
local_model = SentenceTransformerEmbeddingModel(
    model_name="all-MiniLM-L6-v2",  # Lightweight, good performance
    model_batch_size=32,
    device="cuda"  # Use "cpu" if no GPU available
)
```

---

## Customising Embeddings

You can customise the embedding model used at each step by simply creating an instance of the embedding model and then passing it to the relevant function.

Here is how we might do so in the initial clustering step using the Cohere `embed-v4.0` model.

```python
from kura.cluster import generate_base_clusters_from_conversation_summaries
from kura.embedding import CohereEmbeddingModel

# Custom embedding model (Cohere example)
embedding_model = CohereEmbeddingModel(
    model_name="embed-v4.0",
    input_type="clustering"
)

clusters = await generate_base_clusters_from_conversation_summaries(
    summaries,
    embedding_model=embedding_model
)
```

We can also customise the embedding model used in the meta clustering step as seen below.

```python
from kura.meta_cluster import MetaClusterModel

meta_cluster_model = MetaClusterModel(
    model="openai/gpt-4.1-mini",
    embedding_model=embedding_model,  # Same embedding model
    clustering_model=cluster_method
)
```

The default uses `OpenAIEmbeddingModel()` if no model is specified.
