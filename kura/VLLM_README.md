# VLLM Integration for Kura

This document explains how to use VLLM (Very Large Language Models) with Kura for scalable, cost-effective conversation analysis.

## Overview

VLLM integration provides:
- **Cost Efficiency**: 10x+ cost reduction vs APIs for large datasets (100k+ conversations)
- **High Throughput**: 4-8x faster processing through optimized batching
- **Local Processing**: No API dependencies or rate limits
- **GPU Optimization**: Efficient GPU utilization with dynamic batching
- **Scalability**: Support for models from 2B to 70B+ parameters

## Installation

```bash
# Install Kura with VLLM support
pip install -e ".[vllm]"

# Or install VLLM dependencies manually
pip install vllm>=0.5.0 ray>=2.8.0
```

## Hardware Requirements

### Minimum Requirements
- **GPU**: 1x RTX 4090 (24GB) or A100 (40GB)
- **CPU**: 8+ cores
- **RAM**: 32GB+
- **Storage**: 100GB+ for model caching

### Recommended for Production
- **GPU**: 2x A100 (80GB) or 4x A100 (40GB)
- **CPU**: 16+ cores
- **RAM**: 128GB+
- **Storage**: 500GB+ SSD

## Quick Start

### 1. Auto-Configuration

```python
from kura.vllm_config import VLLMConfigManager

# Get recommendations for your scale
manager = VLLMConfigManager()
manager.print_recommendation(n_conversations=50000, task_type="summarization")
manager.print_recommendation(n_conversations=50000, task_type="embedding")
```

### 2. Basic Usage

```python
import asyncio
from kura import Conversation
from kura.vllm_embedding import VLLMEmbeddingModel
from kura.vllm_summarisation import VLLMSummaryModel

async def main():
    # Load conversations
    conversations = Conversation.from_hf_dataset("your-dataset")
    
    # Initialize VLLM models
    embedding_model = VLLMEmbeddingModel(
        model_name="BAAI/bge-large-en-v1.5",
        batch_size=256,
        device="cuda"
    )
    
    summary_model = VLLMSummaryModel(
        model_name="meta-llama/Llama-2-13b-chat-hf",
        tensor_parallel_size=2,
        max_num_seqs=128
    )
    
    # Generate embeddings
    texts = [conv.messages[0].content for conv in conversations[:100]]
    embeddings = await embedding_model.embed(texts)
    
    # Generate summaries
    summaries = await summary_model.summarise(conversations[:100])
    
    print(f"Generated {len(embeddings)} embeddings and {len(summaries)} summaries")

asyncio.run(main())
```

### 3. Integration with Kura Pipeline

```python
from kura.v1 import summarise_conversations, CheckpointManager
from kura.vllm_summarisation import create_vllm_summary_model_for_scale

# Auto-select optimal model for your scale
summary_model = create_vllm_summary_model_for_scale(
    n_conversations=len(conversations),
    prefer_quality=True  # or prefer_speed=True
)

# Use with Kura pipeline
checkpoint_manager = CheckpointManager("./checkpoints", enabled=True)
summaries = await summarise_conversations(
    conversations=conversations,
    model=summary_model,
    checkpoint_manager=checkpoint_manager
)
```

## Configuration Profiles

### Embedding Models

| Scale | Model | Batch Size | Memory | Use Case |
|-------|-------|------------|---------|----------|
| Small (<10k) | all-MiniLM-L6-v2 | 64 | 2GB | Fast prototyping |
| Medium (10k-100k) | bge-base-en-v1.5 | 128 | 4GB | Balanced quality/speed |
| Large (100k+) | bge-large-en-v1.5 | 256 | 8GB | High quality |

### Summarization Models

| Scale | Model | GPUs | Memory | Use Case |
|-------|-------|------|---------|----------|
| Small (<10k) | phi-2 | 1 | 4GB | Fast processing |
| Medium (10k-100k) | Llama-2-7b | 1 | 16GB | Balanced |
| Large (100k-500k) | Llama-2-13b | 2 | 32GB | High quality |
| XL (500k+) | Llama-2-70b | 4 | 160GB | Production scale |

## Performance Optimization

### 1. Batch Size Tuning

```python
from kura.vllm_embedding import BatchOptimizer

optimizer = BatchOptimizer()
optimal_batch_size = optimizer.calculate_optimal_batch_size(
    model_size_gb=1.3,  # bge-large model size
    sequence_length=512,
    model_type="embedding"
)
```

### 2. GPU Memory Optimization

```python
# Optimize for your GPU memory
model = VLLMSummaryModel(
    model_name="meta-llama/Llama-2-13b-chat-hf",
    gpu_memory_utilization=0.9,  # Use 90% of GPU memory
    max_num_seqs=256,  # Adjust based on available memory
    tensor_parallel_size=2  # Use 2 GPUs
)
```

### 3. CPU Fallback

```python
# Automatically fallback to CPU if no GPU available
embedding_model = VLLMEmbeddingModel(
    model_name="all-MiniLM-L6-v2",
    device=None,  # Auto-detect
    num_workers=8  # CPU threads
)
```

## Cost Analysis

### API vs VLLM Cost Comparison

| Dataset Size | API Cost* | VLLM Cost** | Savings | Processing Time |
|--------------|-----------|-------------|---------|-----------------|
| 50k conversations | $500 | $50 | 90% | 4x faster |
| 100k conversations | $1,000 | $75 | 92% | 6x faster |
| 500k conversations | $5,000 | $200 | 96% | 8x faster |

*Estimated OpenAI API costs
**GPU rental costs (A100)

### When to Use VLLM

Use VLLM when:
- ✅ Processing >10k conversations
- ✅ Budget for GPU rental/hardware
- ✅ Need predictable costs
- ✅ Require high throughput
- ✅ Data privacy concerns

Use APIs when:
- ✅ Processing <10k conversations
- ✅ Prototype/one-off analysis
- ✅ No GPU access
- ✅ Need latest models

## Deployment Options

### 1. Local Development

```bash
# Install on local machine with GPU
pip install -e ".[vllm]"
python scripts/tutorial_vllm_api.py
```

### 2. Cloud GPU Instance

```bash
# On AWS/GCP/Azure GPU instance
sudo apt update && sudo apt install -y nvidia-driver-530
pip install -e ".[vllm]"

# Start VLLM server
python -m vllm.entrypoints.api_server \
    --model meta-llama/Llama-2-13b-chat-hf \
    --tensor-parallel-size 2 \
    --max-num-seqs 256
```

### 3. Docker Deployment

```dockerfile
FROM nvidia/cuda:12.1-devel-ubuntu20.04

# Install Python and dependencies
RUN apt update && apt install -y python3 python3-pip git
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install Kura with VLLM
COPY . /app
WORKDIR /app
RUN pip install -e ".[vllm]"

# Start application
CMD ["python", "scripts/tutorial_vllm_api.py"]
```

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**
   ```python
   # Reduce batch size or model size
   model = VLLMSummaryModel(
       model_name="microsoft/phi-2",  # Smaller model
       max_num_seqs=32,  # Smaller batch
       gpu_memory_utilization=0.7  # More conservative
   )
   ```

2. **CUDA Not Available**
   ```bash
   # Check GPU availability
   python -c "import torch; print(torch.cuda.is_available())"
   
   # Install correct CUDA version
   pip install torch==2.0.0+cu121 -f https://download.pytorch.org/whl/torch_stable.html
   ```

3. **Model Loading Failures**
   ```python
   # Use Hugging Face token for private models
   import os
   os.environ["HF_TOKEN"] = "your_token_here"
   ```

### Performance Monitoring

```python
import torch

def monitor_gpu_usage():
    if torch.cuda.is_available():
        memory_used = torch.cuda.memory_allocated() / 1024**3
        memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU Memory: {memory_used:.1f}GB / {memory_total:.1f}GB")

# Monitor during processing
monitor_gpu_usage()
```

## Advanced Features

### 1. Custom Model Fine-tuning

```python
# Use custom fine-tuned models
model = VLLMSummaryModel(
    model_name="/path/to/your/fine-tuned-model",
    tensor_parallel_size=2
)
```

### 2. Multi-GPU Setup

```python
# Automatic multi-GPU scaling
model = VLLMSummaryModel(
    model_name="meta-llama/Llama-2-70b-chat-hf",
    tensor_parallel_size=4,  # Use 4 GPUs
    max_num_seqs=512
)
```

### 3. Production Optimizations

```python
# Production-ready configuration
model = VLLMSummaryModel(
    model_name="meta-llama/Llama-2-13b-chat-hf",
    tensor_parallel_size=2,
    gpu_memory_utilization=0.95,
    max_num_seqs=256,
    enable_prefix_caching=True,  # Cache common prefixes
    trust_remote_code=False  # Security
)
```

## Examples

See `scripts/tutorial_vllm_api.py` for a complete working example that demonstrates:
- Configuration recommendations
- Model initialization
- Integration with Kura pipeline
- Performance monitoring
- Error handling

## Contributing

To contribute to VLLM integration:

1. Test on different GPU configurations
2. Optimize batch sizes for new models
3. Add support for new model architectures
4. Improve error handling and fallbacks
5. Document performance characteristics

## Support

For VLLM-specific issues:
- Check hardware requirements
- Monitor GPU memory usage
- Review VLLM documentation
- Test with smaller models first

For general Kura issues, see the main documentation.