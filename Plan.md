# Plan: Refactoring Clustering to Procedural API

## Current State Analysis

### Existing Class-Based Architecture
The current clustering implementation uses a class-based approach with:

- **`ClusterModel`** (`kura/cluster.py`): Handles embedding summaries, clustering them, and generating cluster descriptions via LLM
- **`MetaClusterModel`** (`kura/meta_cluster.py`): Performs hierarchical clustering to reduce clusters into a tree structure
- **`BaseClusteringMethod`** (`kura/k_means.py`): Low-level clustering algorithms (K-means, etc.)

### Problems with Current Approach
1. **Monolithic classes**: `ClusterModel.cluster_summaries()` does embedding, clustering, and cluster generation in one method
2. **Hidden dependencies**: Models have multiple responsibilities (embedding model, clustering algorithm, LLM client)
3. **Limited composability**: Hard to swap out individual steps or use different models for different steps
4. **Testing complexity**: Difficult to unit test individual operations in isolation

### Success Pattern from Summary Code
The summary implementation (`kura/summarisation.py`) shows good procedural patterns:
- Clear input/output contracts
- Single responsibility per function
- Easy to test and compose
- Supports different models through interfaces

## Proposed Procedural API Design

### High-Level API Functions
```python
# Core clustering functions (similar to v1 summary API)
async def embed_conversation_summaries(
    summaries: List[ConversationSummary],
    *,
    embedding_model: BaseEmbeddingModel,
    checkpoint_manager: Optional[CheckpointManager] = None,
) -> List[EmbeddedSummary]

async def cluster_embedded_summaries(
    embedded_summaries: List[EmbeddedSummary],
    *,
    clustering_method: BaseClusteringMethod,
    checkpoint_manager: Optional[CheckpointManager] = None,
) -> List[SummaryCluster]

async def generate_cluster_descriptions(
    summary_clusters: List[SummaryCluster],
    *,
    llm_client: instructor.AsyncInstructor,
    max_concurrent_requests: int = 50,
    checkpoint_manager: Optional[CheckpointManager] = None,
) -> List[Cluster]

async def reduce_clusters_hierarchically(
    clusters: List[Cluster],
    *,
    embedding_model: BaseEmbeddingModel,
    clustering_method: BaseClusteringMethod,
    llm_client: instructor.AsyncInstructor,
    max_clusters: int = 10,
    max_concurrent_requests: int = 50,
    checkpoint_manager: Optional[CheckpointManager] = None,
) -> List[Cluster]
```

### New Data Types
```python
@dataclass
class EmbeddedSummary:
    """A conversation summary with its embedding vector."""
    summary: ConversationSummary
    embedding: List[float]

@dataclass 
class SummaryCluster:
    """A cluster of summaries before LLM processing."""
    id: int
    summaries: List[ConversationSummary]
    embeddings: List[List[float]]
    contrastive_examples: List[ConversationSummary]
```

### Benefits of This Approach

#### 1. **Composability**
```python
# Users can mix and match components
summaries = await summarise_conversations(conversations, model=openai_summary)
embedded = await embed_conversation_summaries(summaries, embedding_model=openai_embed)
clusters = await cluster_embedded_summaries(embedded, clustering_method=kmeans)
final = await generate_cluster_descriptions(clusters, llm_client=anthropic_client)
```

#### 2. **Flexibility**
```python
# Different embedding models for different steps
base_clusters = await generate_cluster_descriptions(
    clusters, llm_client=openai_client
)
meta_clusters = await reduce_clusters_hierarchically(
    base_clusters, 
    embedding_model=sentence_transformer,  # Different model!
    clustering_method=hdbscan,             # Different algorithm!
    llm_client=anthropic_client           # Different LLM!
)
```

#### 3. **Easier Testing**
```python
# Test individual components in isolation
def test_cluster_embedded_summaries():
    embedded = [EmbeddedSummary(summary=mock_summary, embedding=[0.1, 0.2])]
    result = await cluster_embedded_summaries(embedded, clustering_method=MockCluster())
    assert len(result) == expected_clusters
```

#### 4. **Better Checkpointing**
Each function can have its own checkpoint files:
- `embedded_summaries.jsonl`
- `summary_clusters.jsonl` 
- `cluster_descriptions.jsonl`
- `hierarchical_clusters.jsonl`

#### 5. **Progress Tracking**
Each function handles its own progress reporting, making it clear which step is running.

## Implementation Strategy

### Phase 1: Core Functions
1. **Extract embedding logic** from `ClusterModel._embed_summaries()` into standalone function
2. **Extract clustering logic** from `ClusterModel._generate_clusters_from_embeddings()` 
3. **Extract LLM generation** from `ClusterModel.generate_cluster()` into batch processing function
4. **Create new data types** for intermediate representations

### Phase 2: Meta-Clustering Functions
1. **Break down `MetaClusterModel.reduce_clusters()`** into composable steps:
   - Embedding clusters
   - Clustering the embeddings
   - Generating candidate labels
   - Assigning clusters to labels
   - Creating hierarchical structure

### Phase 3: High-Level Convenience Functions
1. **Composite functions** that combine multiple steps (similar to current v1 API)
2. **Migration utilities** to help users transition from class-based to procedural approach
3. **Backward compatibility** wrappers

### Phase 4: Integration
1. **Update v1 API** to use new procedural functions internally
2. **Update class-based API** to use procedural functions (for backward compatibility)
3. **Documentation and examples**

## Example Usage

### Complete Pipeline
```python
from kura.v2 import (
    embed_conversation_summaries,
    cluster_embedded_summaries, 
    generate_cluster_descriptions,
    reduce_clusters_hierarchically
)

# Step-by-step with full control
embedded = await embed_conversation_summaries(
    summaries, embedding_model=OpenAIEmbeddingModel()
)
summary_clusters = await cluster_embedded_summaries(
    embedded, clustering_method=KmeansClusteringMethod()
)
base_clusters = await generate_cluster_descriptions(
    summary_clusters, llm_client=instructor.from_provider("openai/gpt-4")
)
final_clusters = await reduce_clusters_hierarchically(
    base_clusters,
    embedding_model=SentenceTransformerModel(), 
    clustering_method=HDBSCANClusteringMethod(),
    llm_client=instructor.from_provider("anthropic/claude-3-sonnet"),
    max_clusters=5
)
```

### Convenience Function (for common use cases)
```python
# High-level function combining multiple steps
clusters = await generate_clusters_from_summaries(
    summaries,
    embedding_model=OpenAIEmbeddingModel(),
    clustering_method=KmeansClusteringMethod(clusters_per_group=10),
    llm_client=instructor.from_provider("openai/gpt-4o"),
    max_concurrent_requests=50,
    checkpoint_manager=CheckpointManager("./checkpoints")
)
```

## Migration Path

### Backward Compatibility
Keep existing class-based APIs working by implementing them on top of the new procedural functions:

```python
class ClusterModel(BaseClusterModel):
    async def cluster_summaries(self, summaries: List[ConversationSummary]) -> List[Cluster]:
        # Internally use new procedural API
        embedded = await embed_conversation_summaries(summaries, embedding_model=self.embedding_model)
        clusters = await cluster_embedded_summaries(embedded, clustering_method=self.clustering_method)
        return await generate_cluster_descriptions(clusters, llm_client=self.client)
```

### Gradual Migration
1. **Phase 1**: Introduce procedural functions alongside existing classes
2. **Phase 2**: Update documentation to recommend procedural approach
3. **Phase 3**: Deprecate class-based approach (with migration guide)
4. **Phase 4**: Remove deprecated classes in next major version

## Benefits Summary

1. **Better Testability**: Each function can be unit tested in isolation
2. **Improved Composability**: Mix and match different models/algorithms per step
3. **Clearer Data Flow**: Explicit inputs and outputs for each transformation
4. **Enhanced Flexibility**: Easy to skip steps or add custom processing
5. **Better Error Handling**: Failures are isolated to specific steps
6. **Simpler Debugging**: Clear visibility into each pipeline stage
7. **Functional Programming**: Supports pure functional programming patterns
8. **Heterogeneous Models**: Easy to use different providers for different steps

This procedural approach aligns with the successful pattern established in the summary code and the v1 API, while providing much more granular control over the clustering pipeline.