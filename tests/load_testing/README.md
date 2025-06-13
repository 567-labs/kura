# Kura Load Testing Guide

## Overview

This directory contains load testing tools for Kura's embedding and clustering pipeline. The tests use your existing `logfire_test/clusters.jsonl` data as a baseline and scale it up to measure performance characteristics.

## Quick Start

```bash
# Run basic embedding load test
cd tests/load_testing
python test_embedding_scaling.py

# Or using uv
uv run python test_embedding_scaling.py
```

## What Gets Tested

### 1. Memory Usage
- **Baseline memory**: Memory before processing
- **Peak memory**: Maximum memory during embedding/clustering  
- **Memory scaling**: How memory grows with conversation count
- **Memory efficiency**: Linear vs exponential growth patterns

### 2. Processing Time
- **Embedding time**: Time to generate embeddings via OpenAI API
- **Clustering time**: Time for HDBSCAN clustering
- **Total pipeline time**: End-to-end processing time
- **Throughput**: Conversations processed per minute

### 3. Storage Requirements
- **Embedding storage**: ~6KB per conversation (1536 dims × 4 bytes)
- **Metadata storage**: ~2KB per conversation
- **Total storage**: Combined storage estimates

### 4. API Costs
- **API calls**: Number of OpenAI embedding requests
- **Estimated cost**: Based on OpenAI pricing (~$0.00002/1K tokens)

## Test Configuration

The script uses your existing cluster data from `logfire_test/clusters.jsonl`:

```python
# Default test scales (adjust based on your needs)
scales = [100, 250, 500, 1000, 2000]

# Models used
embedding_model = OpenAIEmbeddingModel(
    model_name="text-embedding-3-small",  # $0.00002/1K tokens
    model_batch_size=50,
    n_concurrent_jobs=5
)
```

## Expected Results

Based on your cluster data (13 clusters, ~900 conversations total):

### Memory Scaling
```
Scale    Peak Memory    Storage
100      ~50MB         ~1MB
500      ~200MB        ~5MB  
1000     ~400MB        ~10MB
2000     ~800MB        ~20MB
```

### Time Scaling  
```
Scale    Time(s)    Throughput(conv/min)
100      ~30s      ~200
500      ~120s     ~250
1000     ~240s     ~250
2000     ~480s     ~250
```

### Cost Scaling
```
Scale    API Calls    Cost
100      ~2          ~$0.0001
500      ~10         ~$0.0005  
1000     ~20         ~$0.001
2000     ~40         ~$0.002
```

## Understanding the Output

### Performance Table
```
Scale | Time (s) | Peak Memory (MB) | Storage (MB) | Throughput (conv/min) | Cost ($)
100   | 32.1     | 48.2            | 1.2          | 187.5                | 0.0001
500   | 118.7    | 186.4           | 6.1          | 252.8                | 0.0006
```

### Scaling Analysis
```
Memory scaling: 3.87x for 5.0x conversations  # Good (sub-linear)
Time scaling: 3.70x for 5.0x conversations    # Good (sub-linear)
Memory efficiency: 1.29 (1.0 = linear)       # Efficient
Time efficiency: 1.35 (1.0 = linear)         # Efficient
```

**Efficiency Interpretation:**
- **1.0**: Perfect linear scaling
- **>1.0**: Better than linear (good)
- **<1.0**: Worse than linear (concerning)

## Customizing Tests

### 1. Adjust Test Scales
```python
# Conservative (for testing)
scales = [50, 100, 200]

# Aggressive (for production planning)  
scales = [100, 500, 1000, 5000, 10000]
```

### 2. Change Embedding Model
```python
# Larger model (higher cost, better quality)
embedding_model = OpenAIEmbeddingModel(
    model_name="text-embedding-3-large",  # 3072 dims, higher cost
    model_batch_size=25,  # Smaller batches for larger model
    n_concurrent_jobs=3
)

# Local model (no API cost)
from kura.embedding import SentenceTransformerEmbeddingModel
embedding_model = SentenceTransformerEmbeddingModel(
    model_name="all-MiniLM-L6-v2",
    model_batch_size=128
)
```

### 3. Mock Mode (No API Calls)
For testing the framework without API costs:

```python
# Edit test_embedding_scaling.py
MOCK_MODE = True  # Set this flag to avoid real API calls
```

## Interpreting Results for Production

### Memory Planning
```python
# For production deployment
conversations_target = 10000
memory_per_1k = peak_memory_mb / (conversations_count / 1000)
required_memory_gb = (conversations_target / 1000) * memory_per_1k / 1024

print(f"For {conversations_target} conversations, need ~{required_memory_gb:.1f}GB RAM")
```

### Cost Planning
```python
# Monthly cost estimation
conversations_per_month = 50000
cost_per_1k = estimated_cost_usd / (conversations_count / 1000)  
monthly_cost = (conversations_per_month / 1000) * cost_per_1k

print(f"Monthly embedding cost: ~${monthly_cost:.2f}")
```

### Performance Planning
```python
# Processing time estimation
conversations_per_batch = 1000
time_per_1k = total_time_s / (conversations_count / 1000)
batch_processing_time = time_per_1k

print(f"Time to process 1K conversations: ~{batch_processing_time:.0f}s")
```

## Troubleshooting

### Out of Memory Errors
```bash
# Reduce test scale or batch size
scales = [50, 100, 200]  # Instead of [100, 500, 1000]

# Or reduce concurrent jobs
n_concurrent_jobs=2  # Instead of 5
```

### API Rate Limiting
```bash
# Reduce concurrency
n_concurrent_jobs=1
model_batch_size=25

# Or add delays between scales
time.sleep(60)  # Wait 1 minute between tests
```

### Unexpected High Costs
```bash
# Enable mock mode first
MOCK_MODE = True

# Or test with smaller scales
scales = [10, 25, 50]
```

## Next Steps

1. **Run the basic test** to establish baselines
2. **Analyze results** for your specific use case  
3. **Plan production resources** based on scaling patterns
4. **Optimize configuration** for your performance/cost targets
5. **Set up monitoring** to track production performance

## Files Generated

- `load_test_results_YYYYMMDD_HHMMSS.json`: Detailed results data
- Memory and performance logs in console output
- Can be imported into spreadsheets for further analysis
