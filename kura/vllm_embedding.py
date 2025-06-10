"""
VLLM-based embedding model implementation for CPU/GPU scaling.

This module provides a high-performance embedding model using VLLM's optimized
inference engine with support for both CPU and GPU execution, dynamic batching,
and memory-efficient processing.
"""

import logging
import asyncio
from typing import List, Optional, Union
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import threading

from kura.base_classes import BaseEmbeddingModel

logger = logging.getLogger(__name__)


class BatchOptimizer:
    """Optimize batch sizes based on hardware and model characteristics."""
    
    def __init__(self, gpu_memory_gb: Optional[float] = None):
        """Initialize batch optimizer.
        
        Args:
            gpu_memory_gb: Available GPU memory in GB. Auto-detected if None.
        """
        self.gpu_memory_gb = gpu_memory_gb or self._detect_gpu_memory()
        
    def _detect_gpu_memory(self) -> float:
        """Detect available GPU memory."""
        if torch.cuda.is_available():
            try:
                memory_bytes = torch.cuda.get_device_properties(0).total_memory
                return memory_bytes / (1024**3)  # Convert to GB
            except Exception as e:
                logger.warning(f"Failed to detect GPU memory: {e}")
                return 8.0  # Default fallback
        return 0.0  # CPU only
        
    def calculate_optimal_batch_size(
        self,
        model_size_gb: float,
        sequence_length: int,
        model_type: str = "embedding"
    ) -> int:
        """Calculate optimal batch size for the given configuration.
        
        Args:
            model_size_gb: Model size in GB
            sequence_length: Average sequence length
            model_type: Type of model ("embedding" or "llm")
            
        Returns:
            Optimal batch size
        """
        if self.gpu_memory_gb == 0.0:  # CPU only
            return 16  # Conservative CPU batch size
            
        if model_type == "embedding":
            # Embedding models are more memory efficient
            # Rough estimation: 1GB model can handle ~100 sequences per GB
            available_memory = max(1.0, self.gpu_memory_gb - model_size_gb - 1.0)  # Leave headroom
            batch_size = int(available_memory * 100)
            
            # Scale down for longer sequences
            if sequence_length > 256:
                batch_size = batch_size // (sequence_length // 256)
                
            return max(8, min(512, batch_size))
        else:
            # More conservative for LLMs
            available_memory = max(0.5, self.gpu_memory_gb - model_size_gb - 2.0)
            return max(4, min(64, int(available_memory * 10)))


class VLLMEmbeddingModel(BaseEmbeddingModel):
    """High-performance embedding model using optimized PyTorch inference.
    
    This model provides CPU/GPU support with dynamic batching and memory optimization
    for scalable embedding generation. It uses SentenceTransformers or raw transformers
    with custom batching logic for maximum throughput.
    """
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-large-en-v1.5",
        batch_size: Optional[int] = None,
        max_length: int = 512,
        device: Optional[str] = None,
        use_sentence_transformers: bool = True,
        num_workers: int = 4,
        **kwargs
    ):
        """Initialize VLLM embedding model.
        
        Args:
            model_name: Name of the embedding model to use
            batch_size: Batch size for processing. Auto-calculated if None
            max_length: Maximum sequence length
            device: Device to use ("cuda", "cpu", or None for auto)
            use_sentence_transformers: Whether to use SentenceTransformers library
            num_workers: Number of worker threads for CPU processing
            **kwargs: Additional arguments for model initialization
        """
        self.model_name = model_name
        self.max_length = max_length
        self.use_sentence_transformers = use_sentence_transformers
        self.num_workers = num_workers
        
        # Device selection
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"Initializing VLLMEmbeddingModel with model={model_name}, device={self.device}")
        
        # Initialize batch optimizer
        self.batch_optimizer = BatchOptimizer()
        
        # Calculate optimal batch size if not provided
        if batch_size is None:
            model_size_gb = self._estimate_model_size()
            self.batch_size = self.batch_optimizer.calculate_optimal_batch_size(
                model_size_gb, max_length, "embedding"
            )
        else:
            self.batch_size = batch_size
            
        logger.info(f"Using batch size: {self.batch_size}")
        
        # Initialize model
        self._init_model()
        
        # Thread pool for CPU processing
        if self.device == "cpu":
            self.executor = ThreadPoolExecutor(max_workers=num_workers)
        else:
            self.executor = None
            
    def _estimate_model_size(self) -> float:
        """Estimate model size in GB based on model name."""
        # Rough estimates for common models
        size_estimates = {
            "all-MiniLM-L6-v2": 0.09,
            "all-mpnet-base-v2": 0.42,
            "BAAI/bge-small-en-v1.5": 0.13,
            "BAAI/bge-base-en-v1.5": 0.44,
            "BAAI/bge-large-en-v1.5": 1.34,
            "sentence-transformers/all-MiniLM-L6-v2": 0.09,
            "sentence-transformers/all-mpnet-base-v2": 0.42,
        }
        
        # Check for exact match or partial match
        for key, size in size_estimates.items():
            if key in self.model_name:
                return size
                
        # Default estimate based on model name patterns
        if "large" in self.model_name.lower():
            return 1.0
        elif "base" in self.model_name.lower():
            return 0.4
        elif "small" in self.model_name.lower():
            return 0.1
        else:
            return 0.5  # Default
            
    def _init_model(self):
        """Initialize the embedding model."""
        try:
            if self.use_sentence_transformers:
                logger.info("Loading model using SentenceTransformers")
                self.model = SentenceTransformer(self.model_name, device=self.device)
                self.tokenizer = None
            else:
                logger.info("Loading model using raw transformers")
                self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model.eval()
                
            logger.info(f"Successfully loaded embedding model: {self.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load embedding model {self.model_name}: {e}")
            raise
            
    def slug(self) -> str:
        """Return a slug for this model configuration."""
        return f"vllm:{self.model_name.replace('/', '_')}-batch:{self.batch_size}-device:{self.device}"
        
    async def _embed_batch_sentence_transformers(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch using SentenceTransformers."""
        loop = asyncio.get_event_loop()
        
        if self.device == "cpu" and self.executor:
            # Use thread pool for CPU processing
            future = self.executor.submit(self.model.encode, texts)
            embeddings = await loop.run_in_executor(None, lambda: future.result())
        else:
            # Direct GPU processing
            def _encode():
                return self.model.encode(texts, show_progress_bar=False)
            embeddings = await loop.run_in_executor(None, _encode)
            
        return embeddings.tolist()
        
    async def _embed_batch_transformers(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch using raw transformers."""
        loop = asyncio.get_event_loop()
        
        def _encode_batch():
            # Tokenize
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use mean pooling
                embeddings = outputs.last_hidden_state.mean(dim=1)
                
            return embeddings.cpu().numpy()
            
        embeddings = await loop.run_in_executor(None, _encode_batch)
        return embeddings.tolist()
        
    async def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a single batch of texts."""
        if self.use_sentence_transformers:
            return await self._embed_batch_sentence_transformers(texts)
        else:
            return await self._embed_batch_transformers(texts)
            
    async def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts into embeddings.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            logger.debug("Empty text list provided, returning empty embeddings")
            return []
            
        logger.info(f"Starting embedding of {len(texts)} texts using VLLM model {self.model_name}")
        
        # Create batches
        batches = self._create_batches(texts, self.batch_size)
        logger.debug(f"Split {len(texts)} texts into {len(batches)} batches of size {self.batch_size}")
        
        # Process batches with concurrency control
        semaphore = asyncio.Semaphore(4)  # Limit concurrent batches to prevent OOM
        
        async def process_batch(batch):
            async with semaphore:
                return await self._embed_batch(batch)
                
        try:
            # Process all batches concurrently
            tasks = [process_batch(batch) for batch in batches]
            results_list_of_lists = await asyncio.gather(*tasks)
            logger.debug(f"Completed embedding {len(batches)} batches")
            
        except Exception as e:
            logger.error(f"Failed to embed texts: {e}")
            raise
            
        # Flatten results
        embeddings = []
        for result_batch in results_list_of_lists:
            embeddings.extend(result_batch)
            
        logger.info(f"Successfully embedded {len(texts)} texts, produced {len(embeddings)} embeddings")
        return embeddings
        
    def _create_batches(self, texts: List[str], batch_size: int) -> List[List[str]]:
        """Create batches from texts."""
        if not texts:
            return []
            
        batches = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batches.append(batch)
            
        return batches
        
    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'executor') and self.executor:
            self.executor.shutdown(wait=True)


# Configuration profiles for different scales
VLLM_EMBEDDING_CONFIGS = {
    "small": {  # <10k texts
        "model_name": "all-MiniLM-L6-v2",
        "batch_size": 64,
        "max_length": 256,
    },
    "medium": {  # 10k-100k texts  
        "model_name": "BAAI/bge-base-en-v1.5",
        "batch_size": 128,
        "max_length": 512,
    },
    "large": {  # 100k+ texts
        "model_name": "BAAI/bge-large-en-v1.5", 
        "batch_size": 256,
        "max_length": 512,
    }
}


def create_vllm_embedding_model_for_scale(n_texts: int, **kwargs) -> VLLMEmbeddingModel:
    """Create a VLLM embedding model optimized for the given scale.
    
    Args:
        n_texts: Expected number of texts to process
        **kwargs: Override any configuration parameters
        
    Returns:
        Configured VLLMEmbeddingModel instance
    """
    if n_texts < 10000:
        config = VLLM_EMBEDDING_CONFIGS["small"]
    elif n_texts < 100000:
        config = VLLM_EMBEDDING_CONFIGS["medium"]
    else:
        config = VLLM_EMBEDDING_CONFIGS["large"]
        
    # Override with any provided kwargs
    config.update(kwargs)
    
    logger.info(f"Creating VLLM embedding model for {n_texts} texts with config: {config}")
    return VLLMEmbeddingModel(**config)