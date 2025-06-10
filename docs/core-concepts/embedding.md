# Embedding

Kura converts text into numerical vectors (embeddings) for clustering and similarity analysis. Embeddings are automatically generated during clustering, but you can configure which model to use.

---

## Embedding Models

Choose between cloud and local models:

```python
from kura.embedding import OpenAIEmbeddingModel, SentenceTransformerEmbeddingModel

# Cloud embeddings (high quality, requires API key)
openai_model = OpenAIEmbeddingModel(
    model_name="text-embedding-3-large",
    model_batch_size=100,
    n_concurrent_jobs=10
)

# Local embeddings (private, runs on your machine)
local_model = SentenceTransformerEmbeddingModel(
    model_name="all-MiniLM-L6-v2",
    model_batch_size=32,
    device="cuda"  # Use "cpu" if no GPU
)
```

---

## Using with Clustering

Embeddings are used automatically in clustering:

```python
from kura import generate_base_clusters_from_conversation_summaries

# Custom embedding model
embedding_model = OpenAIEmbeddingModel(model_name="text-embedding-3-large")

clusters = await generate_base_clusters_from_conversation_summaries(
    summaries,
    embedding_model=embedding_model
)
```

The default uses `OpenAIEmbeddingModel()` if no model is specified.
