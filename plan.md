# Clustering API Refactor: File Structure Implementation

Refactor embedding and clustering to procedural API with clean `<step>.models` structure. There is no need to maintain backwards compatability

## File Structure

```
kura/
├── embedding/
│   ├── __init__.py          # exports embed_summaries()
│   ├── embedding.py         # embed_summaries() implementation
│   └── models/
│       ├── __init__.py      # exports OpenAIEmbeddingModel
│       └── openai.py        # OpenAIEmbeddingModel
├── cluster/
│   ├── __init__.py          # exports cluster_conversations()
│   ├── cluster.py           # cluster_conversations() implementation
│   └── models/
│       ├── __init__.py      # exports KmeansClusteringModel
│       └── kmeans.py        # KmeansClusteringModel
```

## Implementation Checklist

- [ ] **Create `kura/embedding/` directory structure**
- [ ] **Move embedding logic** - Extract `embed_summaries()` to `kura/embedding/embedding.py`
- [ ] **Move `OpenAIEmbeddingModel`** to `kura/embedding/models/openai.py`
- [ ] **Create embedding `__init__.py` files** with proper exports
- [ ] **Create `kura/cluster/` directory structure**
- [ ] **Implement cluster functions** - Extract `cluster_conversations()` with supporting functions
- [ ] **Move `KmeansClusteringMethod`** to `kura/cluster/models/kmeans.py` (rename to `KmeansClusteringModel`)
- [ ] **Create cluster `__init__.py` files** with proper exports
- [ ] **Update base class imports** throughout
- [ ] **Test new import structure** works correctly

## Function Signatures

```python
# Main function - matches summarise_conversations pattern
async def cluster_conversations(
    summaries: list[ConversationSummary],
    *,
    embedding_model: BaseEmbeddingModel,
    clustering_method: BaseClusteringMethod,
    model: str = "openai/gpt-4o-mini",
    contrastive_examples_limit: int = 10,
    max_concurrent_requests: int = 50,
    temperature: float = 0.2,
    checkpoint_manager: Optional[CheckpointManager] = None,
    console: Optional[Console] = None,
    **kwargs,
) -> list[Cluster]:

# Supporting functions
async def embed_summaries(summaries: list[ConversationSummary], embedding_model: BaseEmbeddingModel) -> list[dict[str, Any]]:
def get_contrastive_examples(cluster_id: int, cluster_id_to_summaries: dict, limit: int = 10) -> list[ConversationSummary]:
async def generate_cluster_description(summaries: list[ConversationSummary], contrastive_examples: list[ConversationSummary], model: str, temperature: float = 0.2) -> Cluster:
async def generate_cluster_descriptions(cluster_id_to_summaries: dict, model: str, contrastive_examples_limit: int = 10, max_concurrent_requests: int = 50, temperature: float = 0.2, console: Optional[Console] = None) -> list[Cluster]:
```

## Success Criteria

- [ ] New `cluster_conversations()` produces identical results to old `ClusterModel.cluster_summaries()`
- [ ] All dependencies passed as parameters (no hidden state)
- [ ] Tests pass and verify function behavior
- [ ] API is consistent with `summarise_conversations()` pattern
