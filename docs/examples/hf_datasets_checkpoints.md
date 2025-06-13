# HuggingFace Datasets Checkpoints Guide

This guide demonstrates how to use the new HuggingFace datasets checkpoint system in Kura for improved performance and scalability.

## Quick Start

### Using the Procedural API with HF Datasets

```python
from kura.v1 import (
    summarise_conversations,
    generate_base_clusters_from_conversation_summaries,
    create_hf_checkpoint_manager
)
from kura.types import Conversation
from kura.summarisation import SummaryModel
from kura.cluster import ClusterModel

# Load your conversations
conversations = Conversation.from_hf_dataset("my-dataset/conversations")

# Create HF datasets checkpoint manager
checkpoint_mgr = create_hf_checkpoint_manager(
    checkpoint_dir="./hf_checkpoints",
    hub_repo="my-username/kura-analysis",  # Optional: upload to HF Hub
    compression="gzip"  # Built-in compression
)

# Run pipeline with HF datasets checkpoints
summary_model = SummaryModel()
summaries = await summarise_conversations(
    conversations,
    model=summary_model,
    checkpoint_manager=checkpoint_mgr
)

cluster_model = ClusterModel()
clusters = await generate_base_clusters_from_conversation_summaries(
    summaries,
    model=cluster_model,
    checkpoint_manager=checkpoint_mgr
)
```

### Using Environment Variables

```bash
# Set checkpoint format via environment
export KURA_CHECKPOINT_FORMAT=hf-dataset

# Start Kura server with HF datasets
kura start-app --checkpoint-format hf-dataset
```

## Migration from JSONL

### Analyze Current Checkpoints

```bash
# Analyze existing JSONL checkpoints
kura analyze-checkpoints ./old_checkpoints
```

### Migrate to HF Datasets

```bash
# Basic migration
kura migrate-checkpoints ./old_checkpoints ./new_hf_checkpoints

# Migration with Hub upload
kura migrate-checkpoints ./old_checkpoints ./new_hf_checkpoints \
    --hub-repo my-username/kura-analysis \
    --hub-token $HF_TOKEN \
    --compression gzip
```

## Advanced Features

### Streaming for Large Datasets

```python
# Enable streaming for datasets larger than memory
checkpoint_mgr = create_hf_checkpoint_manager(
    checkpoint_dir="./checkpoints",
    streaming=True  # Process without loading everything into memory
)
```

### Filtering Checkpoints

```python
# Filter clusters without loading all data
large_clusters = checkpoint_mgr.filter_checkpoint(
    "clusters",
    lambda x: len(x["chat_ids"]) > 100,  # Only clusters with >100 conversations
    Cluster
)
```

### Hub Integration

```python
# Automatic backup to HuggingFace Hub
checkpoint_mgr = create_hf_checkpoint_manager(
    checkpoint_dir="./checkpoints",
    hub_repo="my-org/analysis-checkpoints",
    hub_token=os.environ["HF_TOKEN"]
)

# Checkpoints are automatically uploaded and versioned
```

### Checkpoint Information

```python
# Get detailed info about a checkpoint
info = checkpoint_mgr.get_checkpoint_info("summaries")
print(f"Rows: {info['num_rows']}")
print(f"Size: {info['size_bytes']} bytes")
print(f"Columns: {info['column_names']}")
```

## Performance Comparison

| Feature | JSONL Checkpoints | HF Datasets Checkpoints |
|---------|-------------------|--------------------------|
| Memory Usage | Load entire file | Memory-mapped, partial loading |
| Loading Speed | Linear with file size | ~10-100x faster |
| Storage | No compression | 50-80% smaller with compression |
| Querying | Must load all data | Efficient filtering |
| Streaming | Not supported | Process datasets > RAM |
| Versioning | Manual | Built-in via HF Hub |
| Sharing | Manual file transfer | One-click via HF Hub |

## Best Practices

### Choose the Right Format

- **Use JSONL** for:
  - Small datasets (< 10MB)
  - Quick prototyping
  - Legacy compatibility

- **Use HF Datasets** for:
  - Large datasets (> 100MB)
  - Production deployments
  - Collaborative projects
  - Cloud storage needs

### Optimize Performance

```python
# For maximum performance with large datasets
checkpoint_mgr = create_hf_checkpoint_manager(
    checkpoint_dir="./checkpoints",
    streaming=True,           # Don't load everything into memory
    compression="lz4",        # Fast compression
    hub_repo="my-org/data"    # Backup to cloud
)
```

### Handle Schema Evolution

```python
# HF datasets provides automatic schema validation
# If data structure changes, you'll get clear error messages
try:
    data = checkpoint_mgr.load_checkpoint("summaries", ConversationSummary)
except Exception as e:
    print(f"Schema mismatch: {e}")
    # Handle migration or schema update
```

## Troubleshooting

### Common Issues

1. **ImportError: No module named 'datasets'**
   ```bash
   pip install datasets>=3.6.0
   ```

2. **Hub authentication failed**
   ```bash
   # Set up HuggingFace token
   export HF_TOKEN="your_token_here"
   # Or use: huggingface-cli login
   ```

3. **Memory issues with large datasets**
   ```python
   # Enable streaming mode
   checkpoint_mgr = create_hf_checkpoint_manager(streaming=True)
   ```

4. **Slow uploads to Hub**
   ```python
   # Use faster compression
   checkpoint_mgr = create_hf_checkpoint_manager(compression="lz4")
   ```

### Migration Verification

```python
from kura.checkpoints.migration import verify_migration

# Verify migration was successful
results = verify_migration("./old_jsonl", "./new_hf", detailed=True)
print(f"Verified: {results['verified_checkpoints']}/{results['total_checkpoints']}")
```

## Backward Compatibility

The new system is fully backward compatible:

```python
# Existing code continues to work
from kura.v1 import CheckpointManager

# Uses JSONL by default
checkpoint_mgr = CheckpointManager("./checkpoints")

# Opt into HF datasets when ready
checkpoint_mgr = CheckpointManager("./checkpoints", format="hf-dataset")
```

No changes are required to existing code unless you want to use the new features.