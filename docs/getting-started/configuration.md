# Configuration

## Why Configure Kura?

Kura's design lets you easily customise models, prompts and clustering algorithms to tailor the analysis to your specific domain needs.

This is important because to maximise the quality of the generated clusters, it's important to iterate and experiment with different parameters to get the best results.

## CLI Options

Customizing checkpoint storage formats lets you optimize for your dataset size and workflow. HuggingFace format enables streaming for massive datasets, while JSONL is perfect for debugging and smaller analyses.

```bash
kura start-app --dir ./my_checkpoints --checkpoint-format hf-dataset
```

Options: `--dir` (checkpoint path), `--checkpoint-format` (`jsonl` or `hf-dataset`)

## Summary Models

Custom response schemas and prompts extract domain-specific insights that generic summaries miss. Customer support conversations benefit from sentiment analysis, while technical discussions need complexity scoring and solution identification.

```python
summaries = await summarise_conversations(
    conversations,
    model=SummaryModel(model="openai/gpt-4o-mini"),
    response_schema=CustomSummary,  # Your Pydantic schema
    temperature=0.1
)
```

**[→ Summarization Details](../core-concepts/summarization.md)**

## Clustering

Tuning clustering parameters dramatically improves grouping quality for your specific data patterns. Adjusting `clusters_per_group` and embedding models helps create clusters that match your analysis granularity needs.

```python
clusters = await generate_base_clusters_from_conversation_summaries(
    summaries,
    embedding_model=OpenAIEmbeddingModel(model_name="text-embedding-3-small"),
    clustering_method=KmeansClusteringModel(clusters_per_group=15)
)
```

**[→ Clustering Details](../core-concepts/clustering.md)**

## Meta-Clustering

Meta-clustering transforms hundreds of granular clusters into actionable top-level themes. Customizing the reduction ratio helps surface the right level of strategic insights for executive reporting or high-level analysis.

```python
meta_clusters = await reduce_clusters_from_base_clusters(
    clusters,
    model=MetaClusterModel(max_clusters=8)
)
```

**[→ Meta-Clustering Details](../core-concepts/meta-clustering.md)**

## Storage Formats

Choosing the right storage format can reduce processing time by 50% and enable analysis of datasets that wouldn't fit in memory. Parquet offers faster loading and smaller files, while HuggingFace format enables cloud streaming for massive datasets.

```python
# JSONL - debugging and small datasets
checkpoint_mgr = JSONLCheckpointManager("./checkpoints")

# Parquet - 50% smaller files, faster loading
checkpoint_mgr = ParquetCheckpointManager("./checkpoints")

# HuggingFace - streaming and cloud storage
checkpoint_mgr = HFDatasetCheckpointManager("./checkpoints")
```

**[→ Checkpoints Details](../core-concepts/checkpoints.md)**

## Performance Tuning

Optimizing concurrency and caching can reduce analysis time from hours to minutes. Matching request limits to your API tier prevents throttling, while intelligent caching avoids expensive re-computation during iterative development.

```python
summary_model = SummaryModel(
    max_concurrent_requests=25,  # Lower for free tiers
    cache_dir="./.cache"
)
```

**[→ Performance Guide](../guides/performance.md)**
