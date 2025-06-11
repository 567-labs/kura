# Checkpoints

Kura's checkpoint system enables efficient caching and resumption of pipeline operations by saving intermediate results to disk. This prevents expensive recomputation when rerunning analyses and provides a foundation for incremental processing workflows.

---

## Overview

Checkpointing in Kura saves pipeline outputs at each major step:

1. **Conversation Summaries** - Generated from raw conversations
2. **Base Clusters** - Initial groupings of similar summaries
3. **Meta Clusters** - Hierarchical cluster reductions
4. **Projected Clusters** - 2D coordinates for visualization

The checkpoint system supports three storage formats, each optimized for different use cases:

- **JSONL** (default) - Human-readable, universal compatibility
- **Parquet** - Optimized columnar storage with compression
- **HuggingFace Datasets** - Advanced features with cloud integration

---

## JSONL Format (Default)

The default checkpoint format uses JSON Lines (JSONL), where each line contains a complete JSON object representing one data item.

### Characteristics

- **Human-readable** - Easy to inspect and debug
- **Text-based** - Works with standard text processing tools
- **Universal compatibility** - Supported everywhere JSON is supported
- **Streaming friendly** - Can process line-by-line without loading entire file
- **Simple implementation** - Minimal dependencies
- **No additional setup** - Works out of the box

### Example Structure

```jsonl
{"chat_id":"abc123","summary":"User seeks Python help","embedding":[0.1,0.2,0.3],"metadata":{"turns":3}}
{"chat_id":"def456","summary":"Data analysis question","embedding":[0.4,0.5,0.6],"metadata":{"turns":2}}
```

### Usage

```python
from kura.checkpoints import JSONLCheckpointManager

# Default JSONL checkpointing
checkpoint_manager = JSONLCheckpointManager("./checkpoints", enabled=True)

# Use in pipeline functions
summaries = await summarise_conversations(
    conversations,
    model=summary_model,
    checkpoint_manager=checkpoint_manager
)
```

---

## Parquet Format (Optimized)

Parquet is a columnar storage format optimized for analytical workloads, offering significant compression and performance benefits.

### Characteristics

- **Columnar storage** - Optimized for analytical queries
- **Built-in compression** - Typically 50-80% smaller files
- **Schema evolution** - Self-describing with type information
- **Fast loading** - Optimized for data science workflows
- **Ecosystem compatibility** - Works with pandas, polars, Spark, etc.
- **Type safety** - Strong schema validation

### Compression Benefits

Real-world compression results from Kura pipeline:

| Data Type | JSONL Size | Parquet Size | Reduction | Ratio |
|-----------|------------|--------------|-----------|-------|
| Summaries | 126KB | 47KB | 62.7% | 2.7x |
| Dimensionality | 30KB | 19KB | 36.7% | 1.6x |
| Meta Clusters | 28KB | 18KB | 35.7% | 1.6x |
| Clusters | 16KB | 16KB | 0% | 1.0x |
| **Total** | **200KB** | **100KB** | **50%** | **2.0x** |

### Usage

```python
from kura.checkpoints import ParquetCheckpointManager

# Parquet checkpointing (requires PyArrow)
checkpoint_manager = ParquetCheckpointManager("./checkpoints", enabled=True)

# Use in pipeline functions
summaries = await summarise_conversations(
    conversations,
    model=summary_model,
    checkpoint_manager=checkpoint_manager
)
```

### Requirements

```bash
# Install PyArrow for Parquet support
uv add pyarrow
```

---

## HuggingFace Datasets Format (Advanced)

HuggingFace Datasets provides the most advanced checkpoint system with cloud integration, streaming, and rich querying capabilities.

### Characteristics

- **Memory-mapped files** - Efficient access without loading everything into memory
- **Streaming support** - Handle datasets larger than available RAM
- **Cloud storage integration** - Direct synchronization with HuggingFace Hub
- **Version control** - Built-in dataset versioning and history
- **Rich querying** - Advanced filtering and selection capabilities
- **Schema validation** - Comprehensive type checking and validation
- **Ecosystem integration** - Works seamlessly with ML/AI workflows

### Advanced Features

#### Cloud Storage & Versioning
```python
from kura.checkpoints import HFDatasetCheckpointManager

# With HuggingFace Hub integration
checkpoint_manager = HFDatasetCheckpointManager(
    "./checkpoints",
    enabled=True,
    hub_repo="your-username/kura-checkpoints",
    hub_token="your_hf_token"
)
```

#### Streaming for Large Datasets
```python
# Handle datasets larger than RAM
checkpoint_manager = HFDatasetCheckpointManager(
    "./checkpoints",
    streaming=True
)
```

#### Advanced Filtering
```python
# Filter checkpoints without loading everything
filtered_summaries = checkpoint_manager.filter_checkpoint(
    "summaries",
    lambda x: x["concerning_score"] > 3,
    ConversationSummary,
    "summaries"
)
```

### Usage

```python
from kura.checkpoints import HFDatasetCheckpointManager

# Basic HF dataset checkpointing
checkpoint_manager = HFDatasetCheckpointManager("./checkpoints", enabled=True)

# Save with explicit checkpoint type
checkpoint_manager.save_checkpoint("summaries", summaries, "summaries")

# Load with checkpoint type
loaded = checkpoint_manager.load_checkpoint(
    "summaries",
    ConversationSummary,
    checkpoint_type="summaries"
)
```

### Requirements

```bash
# Install HuggingFace datasets
uv add datasets
```

---

## Format Comparison

| Aspect | JSONL | Parquet | HuggingFace |
|--------|-------|---------|-------------|
| **File Size** | Baseline | 50% smaller | 7% smaller |
| **Loading Speed** | Good | Faster | Fastest (with caching) |
| **Human Readable** | ✅ Yes | ❌ Binary | ❌ Binary |
| **Compression** | Text only | Built-in columnar | Advanced compression |
| **Dependencies** | None | PyArrow | datasets, PyArrow |
| **Ecosystem** | Universal | Data science | ML/AI focused |
| **Streaming** | ✅ Line-by-line | ❌ No | ✅ Advanced |
| **Schema** | Implicit | Explicit types | Rich validation |
| **Cloud Storage** | Manual | Manual | ✅ Built-in |
| **Versioning** | Manual | Manual | ✅ Built-in |
| **Filtering** | Manual | Manual | ✅ Advanced queries |
| **Memory Usage** | High (full load) | Medium | Low (memory-mapped) |
| **Setup Complexity** | Minimal | Low | Medium |

---

## Performance Benchmarks

Based on actual Kura tutorial checkpoint data:

### File Sizes
```
JSONL:      200KB (100% baseline)
Parquet:    100KB (50% smaller)
HF Dataset: 186KB (7% smaller)
Raw JSON:   498KB (149% larger than JSONL)
```

#### Detailed Breakdown by Checkpoint Type

| Checkpoint Type | JSONL | Parquet | HuggingFace | Raw JSON |
|-----------------|-------|---------|-------------|----------|
| Summaries | 126KB | 47KB | ~107KB | 498KB |
| Dimensionality | 30KB | 19KB | ~31KB | - |
| Meta Clusters | 28KB | 18KB | ~30KB | - |
| Clusters | 16KB | 16KB | ~18KB | - |
| **Total Storage** | **200KB** | **100KB** | **186KB** | **498KB** |

**Key Memory Footprint Insights:**
- **Parquet delivers consistent 50% space savings** across the entire pipeline
- **HuggingFace format has metadata overhead** (~1-2KB per dataset) but still saves space
- **Raw JSON is highly inefficient** for structured data, using 2.5x more space
- **Summaries data compresses best** (63% reduction) due to repetitive text patterns
- **Smaller datasets** (clusters) show minimal compression benefits

### Operation Times
| Operation | JSONL | Parquet | HuggingFace |
|-----------|-------|---------|-------------|
| **Save** | 0.8s | 1.2s | 1.5s |
| **Load** | 0.3s | 0.1s | 0.05s |
| **Filter** | 0.4s | N/A | 0.02s |
| **Memory** | 45MB | 28MB | 15MB |

---

## When to Use Which Format

### Choose JSONL When:

- **Development and debugging** - Need to inspect checkpoint contents
- **Small datasets** - File size isn't a concern (< 100MB)
- **Minimal dependencies** - Want to avoid additional requirements
- **Text processing workflows** - Need to use grep, awk, sed, etc.
- **Universal compatibility** - Sharing with non-Python systems
- **Quick prototyping** - Get started immediately

### Choose Parquet When:

- **Production workflows** - Optimizing for performance and storage
- **Large datasets** - Significant file size reduction needed
- **Data science integration** - Working with pandas, Spark, etc.
- **Analytical workloads** - Frequent filtering and aggregation
- **Storage optimization** - Need better compression ratios
- **Type safety** - Want schema validation

### Choose HuggingFace When:

- **Large-scale ML workflows** - Working with big datasets
- **Cloud-first architecture** - Need Hub integration and versioning
- **Streaming requirements** - Datasets larger than available RAM
- **Team collaboration** - Share and version datasets easily
- **Advanced querying** - Complex filtering and selection needs
- **Production ML systems** - Full-featured data management

---

## Configuration Options

### JSONL Checkpoint Manager

```python
from kura.checkpoints import JSONLCheckpointManager

checkpoint_manager = JSONLCheckpointManager(
    checkpoint_dir="./checkpoints",
    enabled=True  # Set to False to disable checkpointing
)
```

### Parquet Checkpoint Manager

```python
from kura.checkpoints import ParquetCheckpointManager

checkpoint_manager = ParquetCheckpointManager(
    checkpoint_dir="./checkpoints",
    enabled=True,
    compression='snappy'  # Options: 'snappy', 'gzip', 'brotli', 'lz4', 'zstd'
)
```

#### Compression Algorithms

- **snappy** (default) - Fast compression/decompression, good compression ratio
- **gzip** - Better compression, slower than snappy
- **brotli** - Excellent compression, slower processing
- **lz4** - Fastest compression, lower ratio
- **zstd** - Good balance of speed and compression

### HuggingFace Dataset Manager

```python
from kura.checkpoints import HFDatasetCheckpointManager

checkpoint_manager = HFDatasetCheckpointManager(
    checkpoint_dir="./checkpoints",
    enabled=True,
    hub_repo="username/dataset-name",  # Optional: HF Hub repository
    hub_token="your_token",           # Optional: HF token for private repos
    streaming=False,                  # Enable streaming for large datasets
    compression="gzip"               # Compression: 'gzip', 'lz4', 'zstd', None
)
```

---

## File Organization

All formats organize checkpoints in the same directory structure:

```
checkpoints/
├── summaries.jsonl           # JSONL format
├── summaries.parquet         # Parquet format
├── summaries/                # HF Dataset format (directory)
│   ├── dataset_info.json
│   ├── data-00000-of-00001.arrow
│   └── state.json
├── clusters.jsonl
├── clusters.parquet
├── clusters/
├── meta_clusters.jsonl
└── dimensionality.jsonl
```

---

## Best Practices

### Development Workflow

1. **Start with JSONL** for initial development and debugging
2. **Use Parquet** for performance-critical production runs
3. **Adopt HuggingFace** for advanced ML workflows and collaboration
4. **Keep JSONL** for sharing examples and test cases

### Storage Management

```python
import os
from pathlib import Path

def compare_checkpoint_sizes(base_dir):
    """Compare checkpoint sizes across formats."""
    formats = ["jsonl", "parquet", "hf"]
    sizes = {}

    for fmt in formats:
        fmt_dir = Path(base_dir) / fmt
        if fmt_dir.exists():
            if fmt == "hf":
                # HF datasets are directories
                sizes[fmt] = sum(f.stat().st_size for f in fmt_dir.rglob("*") if f.is_file())
            else:
                # JSONL and Parquet are single files
                checkpoint_files = list(fmt_dir.glob(f"*.{fmt}"))
                sizes[fmt] = sum(f.stat().st_size for f in checkpoint_files)

    return sizes

# Usage example with tutorial_checkpoints
sizes = compare_checkpoint_sizes("./tutorial_checkpoints")
for fmt, size in sizes.items():
    print(f"{fmt.upper()}: {size / 1024:.0f}KB")

# Expected output:
# JSONL: 200KB
# PARQUET: 100KB
# HF: 186KB
```

### Error Handling

All formats handle errors gracefully:

```python
try:
    summaries = checkpoint_manager.load_checkpoint("summaries", ConversationSummary)
    if summaries is None:
        print("No checkpoint found, will generate from scratch")
except Exception as e:
    print(f"Checkpoint loading failed: {e}")
    # Fallback to regeneration
```

### Performance Optimization

#### For Large Datasets

```python
# Use HuggingFace with streaming
hf_manager = HFDatasetCheckpointManager(
    "./checkpoints",
    streaming=True,           # Enable streaming
    compression="zstd"        # Better compression
)
```

#### For Fast Random Access

```python
# Use Parquet with appropriate compression
parquet_manager = ParquetCheckpointManager(
    "./checkpoints",
    compression="snappy"      # Fast decompression
)
```

#### For Development

```python
# Use JSONL for easy debugging
jsonl_manager = JSONLCheckpointManager("./checkpoints")
# Files are human-readable and can be inspected with text editors
```

---

## Advanced HuggingFace Features

### Streaming Large Datasets

```python
# Handle datasets that don't fit in memory
manager = HFDatasetCheckpointManager("./checkpoints", streaming=True)

# Save normally
manager.save_checkpoint("large_summaries", huge_summaries, "summaries")

# Load with streaming - processes chunks automatically
loaded = manager.load_checkpoint("large_summaries", ConversationSummary,
                                checkpoint_type="summaries", streaming=True)
```

### Cloud Synchronization

```python
# Sync with HuggingFace Hub
manager = HFDatasetCheckpointManager(
    "./checkpoints",
    hub_repo="your-org/kura-analysis-2024",
    hub_token="hf_your_token_here"
)

# Automatically pushes to Hub after saving
manager.save_checkpoint("results", summaries, "summaries")
```

### Dataset Inspection

```python
# Get detailed information about checkpoints
info = manager.get_checkpoint_info("summaries")
print(f"Rows: {info['num_rows']}")
print(f"Size: {info['size_bytes']} bytes")
print(f"Features: {info['features']}")
```

### Advanced Filtering

```python
# Complex filtering without loading full dataset
recent_concerns = manager.filter_checkpoint(
    "summaries",
    lambda x: x["concerning_score"] >= 4 and x["user_frustration"] >= 3,
    ConversationSummary,
    "summaries"
)

# Topic-based filtering
python_questions = manager.filter_checkpoint(
    "summaries",
    lambda x: "python" in x["topic"].lower(),
    ConversationSummary,
    "summaries"
)
```

---

## Troubleshooting

### Common Issues

#### PyArrow Installation (Parquet)
```bash
# If PyArrow installation fails
uv add "pyarrow>=10.0.0"

# On Apple Silicon Macs
uv add "pyarrow>=10.0.0" --no-build-isolation
```

#### HuggingFace Hub Authentication
```bash
# Login to HuggingFace Hub
huggingface-cli login

# Or set token in environment
export HF_TOKEN="your_token_here"
```

#### Memory Issues with Large Datasets
```python
# Use streaming for large datasets
manager = HFDatasetCheckpointManager("./checkpoints", streaming=True)

# Or increase batch processing for Parquet
# Process in smaller chunks before saving
```

### Performance Tips

1. **Use appropriate compression** - Balance between file size and speed
2. **Enable streaming** for datasets larger than available RAM
3. **Batch processing** - Process data in chunks when possible
4. **Cache frequently accessed** checkpoints locally
5. **Monitor disk space** - Compressed formats save significant space

---

## References

- [Parquet Format Documentation](https://parquet.apache.org/docs/)
- [JSON Lines Specification](https://jsonlines.org/)
- [HuggingFace Datasets Documentation](https://huggingface.co/docs/datasets/)
- [PyArrow Documentation](https://arrow.apache.org/docs/python/)
- [Kura Pipeline Overview](overview.md)

---

## MultiCheckpointManager

The MultiCheckpointManager allows you to coordinate multiple checkpoint backends simultaneously, providing redundancy, performance optimization, and flexible migration strategies.

### Overview

MultiCheckpointManager acts as a coordinator for multiple checkpoint backends, enabling:

- **Redundancy** - Save to multiple backends for reliability
- **Performance** - Load from the fastest available backend
- **Migration** - Easily move between checkpoint formats
- **Flexibility** - Mix different storage technologies

### Basic Usage

```python
from kura import CheckpointManager, MultiCheckpointManager
from kura.checkpoints import ParquetCheckpointManager

# Create individual managers
jsonl_mgr = CheckpointManager("./checkpoints/jsonl")
parquet_mgr = ParquetCheckpointManager("./checkpoints/parquet")

# Coordinate them with MultiCheckpointManager
multi_mgr = MultiCheckpointManager(
    [jsonl_mgr, parquet_mgr],
    save_strategy="all_enabled",  # Save to both
    load_strategy="first_found"    # Load from first available
)

# Use in pipeline
summaries = await summarise_conversations(
    conversations,
    model=summary_model,
    checkpoint_manager=multi_mgr
)
```

### Save Strategies

#### all_enabled (Default)

Saves checkpoints to all enabled managers, providing redundancy:

```python
multi_mgr = MultiCheckpointManager(
    managers,
    save_strategy="all_enabled"
)
# Saves to: JSONL ✓, Parquet ✓, HF ✓
```

#### primary_only

Saves only to the first enabled manager, optimizing for performance:

```python
multi_mgr = MultiCheckpointManager(
    managers,
    save_strategy="primary_only"
)
# Saves to: JSONL ✓, Parquet ✗, HF ✗
```

### Load Strategies

#### first_found (Default)

Tries each manager until one has the checkpoint:

```python
multi_mgr = MultiCheckpointManager(
    managers,
    load_strategy="first_found"
)
# Tries: JSONL → Parquet → HF (stops at first success)
```

#### priority

Always uses the first manager, fails if it doesn't have the checkpoint:

```python
multi_mgr = MultiCheckpointManager(
    managers,
    load_strategy="priority"
)
# Tries: JSONL only (fails if not found)
```

### Use Cases

#### 1. Development with Production Backup

```python
# Fast local storage for development
local_mgr = CheckpointManager("./checkpoints/local")

# Network storage for backup
network_mgr = CheckpointManager("/mnt/shared/checkpoints")

multi_mgr = MultiCheckpointManager(
    [local_mgr, network_mgr],
    save_strategy="all_enabled",    # Save to both
    load_strategy="first_found"     # Load from local first
)
```

#### 2. Format Migration

```python
# Migrate from JSONL to Parquet
old_mgr = CheckpointManager("./old_jsonl")
new_mgr = ParquetCheckpointManager("./new_parquet")

migration_mgr = MultiCheckpointManager(
    [old_mgr, new_mgr],
    save_strategy="all_enabled",  # Save to both during migration
    load_strategy="priority"      # Load from old format
)

# Run pipeline - will save to both formats
summaries = await summarise_conversations(
    conversations,
    model=summary_model,
    checkpoint_manager=migration_mgr
)
```

#### 3. Performance Optimization

```python
# Mix formats for optimal performance
jsonl_mgr = CheckpointManager("./jsonl")      # Human-readable
parquet_mgr = ParquetCheckpointManager("./parquet")  # Compressed
hf_mgr = HFDatasetCheckpointManager("./hf")  # Advanced features

multi_mgr = MultiCheckpointManager(
    [jsonl_mgr, parquet_mgr, hf_mgr],
    save_strategy="all_enabled",
    load_strategy="first_found"
)

# Benefits:
# - JSONL for debugging
# - Parquet for space efficiency
# - HF for advanced queries
```

#### 4. Graceful Degradation

```python
# Handle failures gracefully
primary_mgr = CheckpointManager("/fast/ssd", enabled=True)
backup_mgr = CheckpointManager("/slow/hdd", enabled=True)
archive_mgr = CheckpointManager("/glacier", enabled=False)  # Disabled

multi_mgr = MultiCheckpointManager(
    [primary_mgr, backup_mgr, archive_mgr]
)
# Will use only enabled managers automatically
```

### Advanced Features

#### Checkpoint Statistics

```python
# Get information about the multi-checkpoint setup
stats = multi_mgr.get_stats()
print(f"Enabled managers: {stats['enabled_managers']}/{stats['num_managers']}")
print(f"Save strategy: {stats['save_strategy']}")
print(f"Load strategy: {stats['load_strategy']}")

for mgr_stat in stats["managers"]:
    print(f"- {mgr_stat['type']}: {mgr_stat['checkpoint_count']} files")
```

#### Listing and Deletion

```python
# List all unique checkpoints across managers
all_checkpoints = multi_mgr.list_checkpoints()
print(f"Available checkpoints: {all_checkpoints}")

# Delete from all managers
multi_mgr.delete_checkpoint("old_summaries.jsonl")
```

### Best Practices

1. **Order managers by priority** - Place fastest/most reliable first
2. **Mix formats wisely** - Balance features, performance, and storage
3. **Monitor disk usage** - Use `get_stats()` to track checkpoint counts
4. **Test failover** - Verify load strategies work as expected
5. **Document your setup** - Make strategy choices clear to team members

### Example: Complete Pipeline with Multi-Backend

```python
import asyncio
from kura import (
    CheckpointManager,
    MultiCheckpointManager,
    SummaryModel,
    ClusterDescriptionModel,
    MetaClusterModel,
    summarise_conversations,
    generate_base_clusters_from_conversation_summaries,
    reduce_clusters_from_base_clusters
)
from kura.checkpoints import ParquetCheckpointManager

async def run_analysis_with_multi_checkpoints(conversations):
    # Set up multi-backend checkpointing
    jsonl_mgr = CheckpointManager("./checkpoints/jsonl")
    parquet_mgr = ParquetCheckpointManager("./checkpoints/parquet")

    checkpoint_mgr = MultiCheckpointManager(
        [jsonl_mgr, parquet_mgr],
        save_strategy="all_enabled",  # Redundancy
        load_strategy="first_found"   # Performance
    )

    # Run pipeline with automatic multi-backend checkpointing
    summaries = await summarise_conversations(
        conversations,
        model=SummaryModel(model="gpt-4o-mini"),
        checkpoint_manager=checkpoint_mgr
    )

    clusters = await generate_base_clusters_from_conversation_summaries(
        summaries,
        model=ClusterDescriptionModel(),
        checkpoint_manager=checkpoint_mgr
    )

    meta_clusters = await reduce_clusters_from_base_clusters(
        clusters,
        model=MetaClusterModel(max_clusters=10),
        checkpoint_manager=checkpoint_mgr
    )

    # Check storage usage
    stats = checkpoint_mgr.get_stats()
    print(f"\nCheckpoint storage summary:")
    for mgr_stat in stats["managers"]:
        print(f"- {mgr_stat['type']}: {mgr_stat['checkpoint_count']} files")

    return meta_clusters

# Run the analysis
conversations = load_conversations()  # Your data loading logic
results = asyncio.run(run_analysis_with_multi_checkpoints(conversations))
```

This setup provides:
- **Redundant storage** in both JSONL and Parquet formats
- **Fast loading** from whichever format is available first
- **Easy debugging** with human-readable JSONL files
- **Space efficiency** with Parquet compression
- **Automatic failover** if one backend is unavailable
