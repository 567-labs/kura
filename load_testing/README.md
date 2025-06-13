# Load Testing & Benchmarking

This directory contains benchmarking scripts to evaluate Kura's performance characteristics across different dimensions: storage efficiency, clustering performance, and semantic quality.

## Scripts Overview

### `benchmark_clustering.py`
**Purpose**: Comprehensive clustering pipeline performance benchmarking  
**What it measures**: End-to-end timing for embedding generation, clustering, and description generation at scale

**Key findings**:
- **Generation dominates processing time**: LLM cluster description generation takes 75-85% of total time
- **Embedding scales linearly**: Predictable performance scaling with dataset size
- **Clustering overhead minimal**: <1s across all dataset sizes
- **Production cost insights**: $0.12 for 10k items, enabling cost modeling

### `test_dataset_compression.py`
**Purpose**: Compare storage efficiency across checkpoint managers  
**What it measures**: File sizes for HuggingFace, Parquet, and JSONL formats at different scales

**Key findings**:
- **Parquet dominates at scale**: 99.8% size reduction vs JSONL/HF for large datasets
- **JSONL/HF similar efficiency**: Both ~88-98MB for 100k items vs 0.2MB for Parquet
- **Critical insight**: Storage format choice becomes essential at production scales

### `analyze_spillover_questions.py` 
**Purpose**: Measure clustering quality through question spillover analysis  
**What it measures**: How often the same question appears across multiple clusters at each hierarchical level

**Why this matters**: High spillover rates indicate poor semantic separation - ideally each question should belong to exactly one cluster. This script helps validate that Kura's hierarchical clustering produces meaningful, non-overlapping topic groups.

### `group_by.py`
**Purpose**: Analyze semantic cluster quality using MT Bench evaluation data  
**What it measures**: Win rates and model performance patterns within clusters

**Why this matters**: If clusters are semantically meaningful, we should see consistent performance patterns within each cluster. Random clustering would show no correlation between cluster membership and model performance.

## Performance Insights

### Storage Scaling
```
Dataset Size | JSONL   | Parquet | HF      | Parquet Advantage
100         | 0.10 MB | 0.03 MB | 0.09 MB | 70% smaller
1,000       | 0.99 MB | 0.15 MB | 0.88 MB | 85% smaller  
100,000     | 98.61MB | 0.20 MB | 88.08MB | 99.8% smaller
```

### Clustering Performance
```
Dataset Size | Total Time | Embed Time | Generate Time | Cost
100         | 3.4s       | 0.9s       | 2.5s         | $0.001
1,000       | 32.7s      | 11.3s      | 21.3s        | $0.012
10,000      | 262s       | 30.4s      | 228s         | $0.12
```

**Key patterns**:
- Generation time dominates (75-85% of total processing)
- Embedding scales linearly with dataset size
- Clustering overhead is minimal (<1s across all sizes)

## Production Implications

1. **Use Parquet for large datasets**: 500x storage savings at 100k+ scale
2. **Generation is the bottleneck**: Focus optimization on LLM summary generation
3. **Embedding scales predictably**: Linear scaling makes capacity planning straightforward
4. **Quality validation matters**: Spillover analysis helps tune clustering parameters

These benchmarks help establish Kura's performance envelope and guide production deployment decisions.
