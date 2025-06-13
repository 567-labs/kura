# Meta-Cluster Refactoring Plan

## Overview

Migrate `kura/meta_cluster.py` from a monolithic 600+ line class-based implementation to a clean, procedural API that follows Kura's architectural principles outlined in AGENT.md.

## Current Problems

1. **Monolithic Design**: 600+ lines mixing orchestration, single-step reduction, and UI logic
2. **Hardcoded Dependencies**: Forces specific embedding/clustering models
3. **Poor Testability**: Dependencies buried in implementation, hard to mock
4. **Mixed Responsibilities**: Progress bars, business logic, and coordination all in one class
5. **Configuration Opacity**: Important parameters hidden in implementation details

## Target Architecture

### Procedural API (Primary Interface)
```python
async def generate_meta_clusters_from_base_clusters(
    clusters: List[Cluster],
    *,
    meta_cluster_model: BaseMetaClusterModel,
    max_clusters: int = 10,
    max_iterations: int = 10,
    checkpoint_manager: Optional[CheckpointManager] = None,
    **kwargs,
) -> List[Cluster]:
```

### Focused Implementation Class
```python
class MetaClusterModel(BaseMetaClusterModel):
    def __init__(
        self,
        model: Union[str, KnownModelName] = "openai/gpt-4o-mini",
        max_concurrent_requests: int = 50,
        temperature: float = 0.2,
        embedding_model: BaseEmbeddingModel = OpenAIEmbeddingModel(),
        clustering_method: BaseClusteringMethod = KmeansClusteringModel(12),
        checkpoint_filename: str = "meta_clusters.jsonl",
        console: Optional[Console] = None,
    )
    
    async def reduce_clusters(self, clusters: List[Cluster], **kwargs) -> List[Cluster]:
        """Single-step cluster reduction only"""
```

## Migration Strategy

### Phase 1: Foundation & Core Logic
1. **Extract Core Business Logic**: Identify and separate the essential meta-clustering algorithms
2. **Create New Implementation**: Build new `MetaClusterModel` with dependency injection
3. **Implement Single-Step Reduction**: Focus only on one iteration of cluster reduction
4. **Add Basic Tests**: Ensure core functionality works in isolation

### Phase 2: Procedural API & Orchestration
1. **Implement Orchestration Function**: Create `generate_meta_clusters_from_base_clusters()`
2. **Add Iteration Logic**: Handle multiple reduction iterations until target cluster count
3. **Integrate Checkpointing**: Add checkpoint manager support for resumable operations
4. **Progress Reporting**: Add optional progress callbacks, separate from business logic

### Phase 3: Integration & Migration
1. **Update Existing Callers**: Modify code that uses old `MetaClusterModel`
2. **Maintain Backward Compatibility**: Create adapter or deprecation path
3. **UI/Progress Integration**: Ensure Rich progress bars still work via callbacks
4. **Performance Testing**: Verify new implementation maintains or improves performance

### Phase 4: Cleanup & Documentation
1. **Remove Old Implementation**: Delete legacy code after migration complete
2. **Update Documentation**: Reflect new procedural API design
3. **Add Usage Examples**: Show both simple and advanced usage patterns
4. **Integration Tests**: Ensure end-to-end workflows still function

## Key Design Principles

### 1. Dependency Injection Over Hardcoding
```python
# ✅ Good: Configurable dependencies
model = MetaClusterModel(
    embedding_model=custom_embedding_model,
    clustering_method=custom_clustering_method,
    temperature=0.1,
    max_concurrent_requests=100
)

# ❌ Bad: Hardcoded dependencies (current)
model = MetaClusterModel()  # Uses hardcoded OpenAI + K-means
```

### 2. Single Responsibility Classes
- `MetaClusterModel`: Only does single-step cluster reduction
- Orchestration function: Handles iteration logic and coordination
- Progress reporting: Separate concern via callbacks

### 3. Configuration Transparency
- All important parameters exposed in function/constructor signatures
- No configuration buried in implementation details
- Easy to understand what can be customized

### 4. Testability First
- Dependencies can be easily mocked
- Individual components testable in isolation
- Pure functions where possible

## Implementation Details

### Core Methods to Extract
From current implementation, these methods contain the essential logic:
- `generate_candidate_clusters()` - LLM call to create meta-cluster names
- `label_cluster()` - Assign clusters to meta-cluster categories  
- `rename_cluster_group()` - Generate final meta-cluster names/descriptions

### Progress Reporting Strategy
Instead of embedding Rich progress bars in business logic:
```python
# Progress callback approach
async def generate_meta_clusters_from_base_clusters(
    clusters: List[Cluster],
    *,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
    **kwargs
) -> List[Cluster]:
    if progress_callback:
        progress_callback("Embedding clusters", 0, len(clusters))
    # ... business logic ...
    if progress_callback:
        progress_callback("Clustering", current_step, total_steps)
```

### Error Handling
- Specific exception types for different failure modes
- Retry logic configurable via parameters
- Graceful degradation when possible

## Success Criteria

1. **Functionality Preserved**: All existing meta-clustering behavior works
2. **Improved Testability**: Can mock all dependencies, test components in isolation
3. **Better Configuration**: All important parameters exposed and configurable
4. **Cleaner Architecture**: Clear separation between orchestration and implementation
5. **Performance Maintained**: No regression in speed or resource usage
6. **Backward Compatibility**: Existing code continues to work (temporarily)

## Risk Mitigation

1. **Incremental Migration**: Build new alongside old, migrate callers gradually
2. **Comprehensive Testing**: Ensure behavior equivalence before switching
3. **Feature Flags**: Allow runtime switching between old/new implementations
4. **Documentation**: Clear migration guide for users of the API

## Timeline Estimates

- **Phase 1**: 2-3 days (core logic extraction and new implementation)
- **Phase 2**: 1-2 days (procedural API and orchestration)  
- **Phase 3**: 1-2 days (integration and migration)
- **Phase 4**: 1 day (cleanup and documentation)

**Total**: ~5-8 days of focused development time
