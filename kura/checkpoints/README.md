# Kura Checkpoint System

This directory contains the checkpoint management system for Kura, providing efficient caching and resumption of pipeline operations by saving intermediate results to disk.

## Architecture Overview

The checkpoint system supports multiple storage backends, each optimized for different use cases:

```
kura/checkpoints/
├── __init__.py          # Module exports and availability detection
├── jsonl.py             # JSONL checkpoint manager (default)
├── parquet.py           # Parquet checkpoint manager (optimized)
├── hf_dataset.py        # HuggingFace datasets manager (advanced)
└── README.md           # This file
```

## Checkpoint Managers

### JSONLCheckpointManager (Default)
- **Format**: JSON Lines (human-readable text)
- **Dependencies**: None
- **Use Case**: Development, debugging, small datasets
- **Pros**: Human-readable, universal compatibility, simple
- **Cons**: Larger file sizes, slower for large datasets

### ParquetCheckpointManager (Optimized)
- **Format**: Parquet (columnar binary)
- **Dependencies**: PyArrow
- **Use Case**: Production analytics, data science workflows
- **Pros**: 50% smaller files, fast loading, type safety
- **Cons**: Binary format, requires PyArrow

### HFDatasetCheckpointManager (Advanced)
- **Format**: HuggingFace Datasets (Arrow/Parquet)
- **Dependencies**: datasets, PyArrow
- **Use Case**: Large-scale ML, cloud workflows, team collaboration
- **Pros**: Streaming, cloud sync, versioning, advanced querying
- **Cons**: Most complex, highest dependencies

## Quick Start

### Basic Usage

```python
from kura.checkpoints import JSONLCheckpointManager, ParquetCheckpointManager

# JSONL (always available)
manager = JSONLCheckpointManager("./checkpoints")
manager.save_checkpoint("summaries.jsonl", summaries)
loaded = manager.load_checkpoint("summaries.jsonl", ConversationSummary)

# Parquet (if PyArrow available)
manager = ParquetCheckpointManager("./checkpoints")
manager.save_checkpoint("summaries.parquet", summaries)
loaded = manager.load_checkpoint("summaries.parquet", ConversationSummary)

# HuggingFace (if datasets available)
manager = HFDatasetCheckpointManager("./checkpoints")
manager.save_checkpoint("summaries", summaries, "summaries")
loaded = manager.load_checkpoint("summaries", ConversationSummary, checkpoint_type="summaries")
```

### Memory Footprint Comparison

Based on real Kura tutorial checkpoint data:

| Checkpoint Type | JSONL | Parquet | HuggingFace | Raw JSON |
|-----------------|-------|---------|-------------|----------|
| Summaries | 126KB | 47KB | ~107KB | 498KB |
| Dimensionality | 30KB | 19KB | ~31KB | - |
| Meta Clusters | 28KB | 18KB | ~30KB | - |
| Clusters | 16KB | 16KB | ~18KB | - |
| **Total Storage** | **200KB** | **100KB** | **186KB** | **498KB** |
| **Space Savings** | Baseline | 50% smaller | 7% smaller | 149% larger |

**Key Insights:**
- **Parquet consistently saves 50% space** across the pipeline
- **HuggingFace has metadata overhead** but still compresses well
- **Summaries compress best** (63% reduction) due to text patterns
- **Raw JSON is highly inefficient** for structured data

### Checking Availability

```python
from kura.checkpoints import PARQUET_AVAILABLE, HF_DATASETS_AVAILABLE

if PARQUET_AVAILABLE:
    # Use ParquetCheckpointManager
    pass

if HF_DATASETS_AVAILABLE:
    # Use HFDatasetCheckpointManager
    pass
```

## Testing

The checkpoint system uses **unified testing** with `pytest.mark.parametrize` to ensure consistent behavior across all managers.

### Running Tests

```bash
# Run unified tests across all checkpoint managers
uv run pytest tests/test_checkpoints_unified.py -v
```

### Test Structure

The unified test suite (`tests/test_checkpoints_unified.py`) runs the same tests across all available checkpoint managers:

```python
@pytest.mark.parametrize("manager_name,manager_class,manager_kwargs", create_manager_params())
class TestCheckpointManagerUnified:
    def test_summaries_roundtrip(self, manager_name, manager_class, manager_kwargs, sample_summaries):
        # This test runs on JSONL, Parquet, AND HuggingFace managers
```

This approach ensures:
- **Behavioral Consistency**: All managers handle the same data identically
- **Automatic Coverage**: New tests automatically run on all managers
- **Clear Failure Reporting**: `test_name[manager_name]` shows exactly which manager fails

## Contributing

### Adding a New Checkpoint Manager

1. **Create the Manager Class**:

```python
# kura/checkpoints/my_format.py
class MyFormatCheckpointManager:
    def __init__(self, checkpoint_dir: str, *, enabled: bool = True):
        self.checkpoint_dir = checkpoint_dir
        self.enabled = enabled

    def save_checkpoint(self, filename: str, data: List[T]) -> None:
        # Implementation
        pass

    def load_checkpoint(self, filename: str, model_class: type[T]) -> Optional[List[T]]:
        # Implementation
        pass

    def list_checkpoints(self) -> List[str]:
        # Implementation
        pass
```

2. **Update Module Exports**:

```python
# kura/checkpoints/__init__.py
try:
    from .my_format import MyFormatCheckpointManager
    MY_FORMAT_AVAILABLE = True
except ImportError:
    MyFormatCheckpointManager = None
    MY_FORMAT_AVAILABLE = False

__all__ = [
    "JSONLCheckpointManager",
    "ParquetCheckpointManager",
    "HFDatasetCheckpointManager",
]

if MY_FORMAT_AVAILABLE:
    __all__.append("MyFormatCheckpointManager")
```

3. **Add to Unified Tests**:

```python
# tests/test_checkpoints_unified.py
def create_manager_params():
    params = [
        # ... existing managers
    ]

    # Add your manager if available
    if MY_FORMAT_AVAILABLE:
        params.append(pytest.param(
            "my_format",
            MyFormatCheckpointManager,
            {},  # kwargs
            id="my_format"
        ))

    return params
```

4. **Handle Manager-Specific Parameters**:

```python
# tests/test_checkpoints_unified.py
def get_checkpoint_params(manager_name: str, filename: str, data: List, model_class):
    if manager_name == "my_format":
        # Return appropriate save/load parameters for your manager
        return {
            "save_params": (filename, data),
            "load_params": (filename, model_class),
            "load_kwargs": {}
        }
    # ... handle other managers
```

### Design Principles

When implementing a new checkpoint manager, follow these principles:

#### 1. **Consistent Interface**
All managers should implement the same core methods:
- `save_checkpoint(filename, data)`
- `load_checkpoint(filename, model_class) -> Optional[List[T]]`
- `list_checkpoints() -> List[str]`

#### 2. **Graceful Degradation**
- Handle missing dependencies gracefully
- Use try/catch imports with availability flags
- Return `None` for missing checkpoints (not empty lists)

#### 3. **Pydantic Integration**
- Accept any `BaseModel` subclass
- Preserve all field values during roundtrip
- Handle optional fields correctly (`None` vs empty values)

#### 4. **Error Handling**
```python
def load_checkpoint(self, filename: str, model_class: type[T]) -> Optional[List[T]]:
    if not self.enabled:
        return None

    try:
        # Load implementation
        pass
    except Exception as e:
        logger.error(f"Failed to load checkpoint {filename}: {e}")
        return None
```

#### 5. **Logging**
Use consistent logging patterns:
```python
import logging
logger = logging.getLogger(__name__)

logger.info(f"Saved checkpoint to {path} with {len(data)} items")
logger.error(f"Failed to load checkpoint from {path}: {e}")
```

### Testing Guidelines

#### 1. **Data Types to Test**
Ensure your manager handles all Kura data types:
- `Conversation` - Complex nested structure with messages
- `ConversationSummary` - Optional fields, embeddings, metadata
- `Cluster` - Simple structure with relationships
- `ProjectedCluster` - Extends Cluster with coordinates

#### 2. **Edge Cases to Handle**
- Empty data lists
- Nonexistent files
- Disabled managers
- Optional fields set to `None`
- Float precision (embeddings, coordinates)
- Complex metadata structures

#### 3. **Performance Considerations**
- Test with reasonably sized datasets (100+ items)
- Measure file sizes for compression claims
- Consider memory usage for large datasets

## Common Issues & Troubleshooting

### Import Errors

**PyArrow Missing**:
```bash
uv add pyarrow
# or on Apple Silicon:
uv add "pyarrow>=10.0.0" --no-build-isolation
```

**HuggingFace Datasets Missing**:
```bash
uv add datasets
```

### Test Failures

**Float Precision Issues**:
```python
# Use tolerance for floating point comparisons
assert abs(original.x_coord - loaded.x_coord) < 1e-5
```

**Empty Data Handling**:
```python
# All managers should return None for empty/missing data
assert manager.load_checkpoint("empty.jsonl", ConversationSummary) is None
```

**Timestamp Handling**:
```python
# Handle different timestamp representations
if isinstance(timestamp, str):
    timestamp = datetime.fromisoformat(timestamp)
elif hasattr(timestamp, 'timestamp'):
    timestamp = timestamp.to_pydatetime()
```

### Performance Issues

**Large Files**:
- Use Parquet for files > 100MB (50% space savings)
- Use HuggingFace streaming for files > 1GB
- Consider chunked processing

**Memory Usage**:
- HuggingFace: Memory-mapped files (~7% space savings)
- Parquet: Columnar efficiency (50% space savings)
- JSONL: Line-by-line streaming (baseline)
- Raw JSON: Avoid for structured data (149% larger)

## Related Documentation

- [Core Concepts: Checkpoints](../../docs/core-concepts/checkpoints.md) - User guide
- [Parquet Format](https://parquet.apache.org/docs/) - Parquet documentation
- [HuggingFace Datasets](https://huggingface.co/docs/datasets/) - HF datasets documentation
- [PyArrow](https://arrow.apache.org/docs/python/) - PyArrow documentation
