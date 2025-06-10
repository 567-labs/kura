# Meta-Clustering Refactor Plan

## Overview
Refactor `kura/meta_cluster.py` to follow the procedural API patterns established in `cluster.py` and `summarisation.py`. The goal is to create composable, configurable functions with clear separation of concerns.

## 1. Core Function Signatures

### Main Procedural Function
```python
async def generate_meta_clusters_from_base_clusters(
    clusters: List[Cluster],
    *,
    meta_cluster_model: BaseMetaClusterModel,
    max_clusters: int = 10,
    max_iterations: int = 10,
    checkpoint_manager: Optional[CheckpointManager] = None,
    **kwargs
) -> List[Cluster]:
    """
    Generate hierarchical meta-clusters from base clusters.
    
    Iteratively calls meta_cluster_model.reduce_clusters() until the number 
    of root clusters is <= max_clusters or max_iterations is reached.
    
    Args:
        clusters: Base clusters to meta-cluster
        meta_cluster_model: Model that performs single-step cluster reduction
        max_clusters: Maximum number of root clusters to produce
        max_iterations: Maximum number of reduction iterations
        checkpoint_manager: Optional checkpoint manager for caching
        
    Returns:
        List of hierarchical clusters with parent-child relationships
    """
```

### Simple Orchestration Function
The main function is now just simple orchestration:

```python
async def generate_meta_clusters_from_base_clusters(
    clusters: List[Cluster],
    *,
    meta_cluster_model: BaseMetaClusterModel,
    max_clusters: int = 10,
    max_iterations: int = 10,
    checkpoint_manager: Optional[CheckpointManager] = None,
    **kwargs
) -> List[Cluster]:
    """Pure orchestration - just calls reduce_clusters repeatedly."""
    
    # Load from checkpoint if available
    if checkpoint_manager:
        cached = checkpoint_manager.load_checkpoint(meta_cluster_model.checkpoint_filename, Cluster)
        if cached:
            return cached
    
    # Initialize tracking
    all_clusters = clusters.copy()
    current_roots = clusters.copy()
    
    # Iterative reduction loop
    for iteration in range(max_iterations):
        if len(current_roots) <= max_clusters:
            break
            
        logger.info(f"Meta-clustering iteration {iteration + 1}: {len(current_roots)} → target: {max_clusters}")
        
        # Single iteration of meta-clustering
        iteration_result = await meta_cluster_model.reduce_clusters(current_roots, **kwargs)
        
        # Update tracking
        new_roots = [c for c in iteration_result if c.parent_id is None]
        old_cluster_ids = {c.id for c in iteration_result if c.parent_id is not None}
        
        # Remove old clusters that now have parents
        all_clusters = [c for c in all_clusters if c.id not in old_cluster_ids]
        all_clusters.extend(iteration_result)
        
        current_roots = new_roots
    
    # Save to checkpoint
    if checkpoint_manager:
        checkpoint_manager.save_checkpoint(meta_cluster_model.checkpoint_filename, all_clusters)
    
    return all_clusters
```

### Simplified MetaClusterModel
```python
class MetaClusterModel(BaseMetaClusterModel):
    """
    Model for performing a single iteration of meta-clustering.
    
    Takes a list of clusters and reduces them by one level, creating
    meta-clusters with hierarchical parent-child relationships.
    """
    
    def __init__(
        self,
        model: Union[str, KnownModelName] = "openai/gpt-4o-mini",
        max_concurrent_requests: int = 50,
        temperature: float = 0.2,
        embedding_model: BaseEmbeddingModel = OpenAIEmbeddingModel(),
        clustering_method: BaseClusteringMethod = KmeansClusteringModel(12),
        checkpoint_filename: str = "meta_clusters.jsonl",
        console: Optional[Console] = None,
    ):
        """Initialize with all dependencies for single-step reduction."""
        
    async def reduce_clusters(
        self,
        clusters: List[Cluster],
        **kwargs
    ) -> List[Cluster]:
        """
        Perform a single iteration of meta-clustering.
        
        Steps:
        1. Embed clusters using embedding_model
        2. Group similar clusters using clustering_method
        3. Generate candidate meta-cluster names via LLM
        4. Label each cluster group with appropriate candidate  
        5. Create hierarchical structure with parent meta-clusters
        
        Args:
            clusters: Clusters to reduce in this iteration
            
        Returns:
            List of clusters with new parent clusters created
        """
```

## 2. Extension Examples

### Custom Prompts
```python
# Custom candidate generation prompt
CUSTOM_CANDIDATE_PROMPT = """
Create higher-level categories that focus on technical complexity.
Prioritize groupings by programming language and difficulty level.
"""

# Usage
meta_clusters = await generate_meta_clusters_from_base_clusters(
    clusters=base_clusters,
    meta_cluster_model=MetaClusterModel(
        model="openai/gpt-4",
        temperature=0.1,
    ),
    max_clusters=5,
)
```

### Custom Models
```python
# Different embedding model for semantic clustering
from kura.embedding import HuggingFaceEmbeddingModel

# Different clustering algorithm
from kura.cluster import HDBSCANClusteringModel

meta_clusters = await generate_meta_clusters_from_base_clusters(
    clusters=base_clusters,
    meta_cluster_model=MetaClusterModel(
        model="anthropic/claude-3-sonnet",
        embedding_model=HuggingFaceEmbeddingModel(model="sentence-transformers/all-MiniLM-L6-v2"),
        clustering_method=HDBSCANClusteringModel(min_cluster_size=3),
    ),
    max_clusters=5,
)
```

### Custom Meta-Cluster Model
```python
class CustomMetaClusterModel(BaseMetaClusterModel):
    """Custom meta-clustering with domain-specific logic."""
    
    def __init__(
        self,
        model: str = "openai/gpt-4o-mini",
        embedding_model: BaseEmbeddingModel = OpenAIEmbeddingModel(),
        clustering_method: BaseClusteringMethod = KmeansClusteringModel(12),
        **kwargs
    ):
        # Custom initialization logic
        super().__init__(model=model, embedding_model=embedding_model, clustering_method=clustering_method, **kwargs)
        
    async def reduce_clusters(self, clusters: List[Cluster], **kwargs) -> List[Cluster]:
        # Custom single-iteration logic
        # Maybe use different prompts based on cluster content
        # Or apply business rules for grouping
        
        # Could call super().reduce_clusters() and post-process
        # Or implement completely custom logic
        pass

# Usage
custom_model = CustomMetaClusterModel(
    embedding_model=HuggingFaceEmbeddingModel("sentence-transformers/all-MiniLM-L6-v2"),
    clustering_method=HDBSCANClusteringModel(min_cluster_size=5),
)
meta_clusters = await generate_meta_clusters_from_base_clusters(
    clusters=base_clusters,
    meta_cluster_model=custom_model,
)
```

### Progress Tracking
```python
from rich.console import Console

console = Console()
meta_clusters = await generate_meta_clusters_from_base_clusters(
    clusters=base_clusters,
    meta_cluster_model=MetaClusterModel(console=console),  # Progress tracking built into model
)
```

## 3. Implementation Benefits

### Simplicity
- **Procedural function**: Pure orchestration (iteration loop, checkpointing)
- **MetaClusterModel**: Encapsulates single-step reduction with all dependencies
- **Clean interface**: Only one method (`reduce_clusters`) needs to be implemented

### Composability
- Users can mix and match embedding models, clustering algorithms, and LLM models
- All dependencies injected via constructor
- Easy to test individual components in isolation

### Configurability  
- All important parameters exposed in constructor or function arguments
- Custom prompts, temperatures, and model settings configurable
- Progress display and checkpointing optional

### Extensibility
- Custom models implement single `reduce_clusters` method
- Custom embedding/clustering algorithms pluggable via constructor
- Business logic can be added by overriding `reduce_clusters`

### Testability
- `generate_meta_clusters_from_base_clusters()` is pure orchestration - easy to test
- `MetaClusterModel.reduce_clusters()` can be mocked for testing iteration logic
- Individual components (embedding, clustering, LLM) can be tested separately

## 4. Migration Path

1. **Phase 1**: Create new `MetaClusterModel` class with `reduce_clusters()` method containing all logic
2. **Phase 2**: Implement main `generate_meta_clusters_from_base_clusters()` procedural function
3. **Phase 3**: Update existing code to use new procedural API
4. **Phase 4**: Remove old complex methods from existing `MetaClusterModel`
5. **Phase 5**: Clean up any remaining UI logic that should be in separate modules
