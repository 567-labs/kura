# Configuration

This guide explains the various configuration options available in Kura using its procedural API (v1). This API is best for flexible pipelines where you need fine control over individual steps, the ability to skip or reorder steps, A/B test different models, or prefer a functional programming style.

## Checkpoint Files

Kura saves several checkpoint files during processing:

| Checkpoint File        | Description                      |
| ---------------------- | -------------------------------- |
| `conversations.json`   | Raw conversation data            |
| `summaries.jsonl`      | Summarized conversations         |
| `clusters.jsonl`       | Base cluster data                |
| `meta_clusters.jsonl`  | Hierarchical cluster data        |
| `dimensionality.jsonl` | Projected data for visualization |

Checkpoint filenames are now defined as properties in their respective model classes. When using the procedural API, checkpoint management is handled via the `CheckpointManager`.

## CLI Configuration

When using the CLI, you can configure the checkpoint directory:

```bash
# Start the web server with a custom checkpoint directory
kura --dir ./my_checkpoints
```

The procedural API provides flexibility by breaking the pipeline into composable functions:

```python
from kura import (
    summarise_conversations,
    generate_base_clusters_from_conversation_summaries,
    reduce_clusters_from_base_clusters,
    reduce_dimensionality_from_clusters,
    CheckpointManager
)
from kura.summarisation import SummaryModel
from kura.cluster import ClusterModel
from kura.meta_cluster import MetaClusterModel
from kura.dimensionality import HDBUMAP
# Assuming Conversation type might be needed for context, if not, it can be removed.
# from kura.types import Conversation

# Sample conversations (replace with your actual data loading)
# conversations = [Conversation(...)]

# Configure models independently
summary_model = SummaryModel()
cluster_model = ClusterModel()
meta_cluster_model = MetaClusterModel(max_clusters=10)
dimensionality_model = HDBUMAP()

# Optional checkpoint management
checkpoint_manager = CheckpointManager("./my_checkpoints", enabled=True)

# Run pipeline with keyword arguments
async def analyze(conversations): # Added conversations as an argument
    summaries = await summarise_conversations(
        conversations,
        model=summary_model,
        checkpoint_manager=checkpoint_manager
    )

    clusters = await generate_base_clusters_from_conversation_summaries(
        summaries,
        model=cluster_model,
        checkpoint_manager=checkpoint_manager
    )

    reduced = await reduce_clusters_from_base_clusters(
        clusters,
        model=meta_cluster_model,
        checkpoint_manager=checkpoint_manager
    )

    projected = await reduce_dimensionality_from_clusters(
        reduced,
        model=dimensionality_model,
        checkpoint_manager=checkpoint_manager
    )

    return projected
```

The procedural API excels at working with different model implementations for the same task:

```python
# Use different backends for the same task
from kura import summarise_conversations
# Assuming these model classes exist and are correctly imported
# from kura.summarisation import OpenAISummaryModel, VLLMSummaryModel, HuggingFaceSummaryModel

# Sample conversations (replace with your actual data loading)
# conversations = [...]
# checkpoint_mgr = CheckpointManager("./my_checkpoints")

# OpenAI backend
# openai_summaries = await summarise_conversations(
#     conversations,
#     model=OpenAISummaryModel(api_key="sk-..."), # Replace with actual model init if different
#     checkpoint_manager=checkpoint_mgr
# )

# Local vLLM backend
# vllm_summaries = await summarise_conversations(
#     conversations,
#     model=VLLMSummaryModel(model_path="/models/llama"), # Replace with actual model init if different
#     checkpoint_manager=checkpoint_mgr
# )

# Hugging Face backend
# hf_summaries = await summarise_conversations(
#     conversations,
#     model=HuggingFaceSummaryModel("facebook/bart-large-cnn"), # Replace with actual model init if different
#     checkpoint_manager=checkpoint_mgr
# )
```

_Note: The heterogeneous models example has been commented out as it relies on specific model classes (`OpenAISummaryModel`, `VLLMSummaryModel`, `HuggingFaceSummaryModel`) whose existence and import paths are not confirmed from the provided context. Ensure these are correctly defined and imported in your actual usage._

## Next Steps

Now that you understand how to configure Kura using the procedural API, you can:

- [Learn about core concepts](../core-concepts/overview.md)
- [Try the Procedural API Tutorial](../getting-started/quickstart.md)
- [Check out the API Reference](../api/index.md)
