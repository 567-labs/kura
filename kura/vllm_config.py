"""
VLLM configuration utilities for optimal model selection and performance tuning.

This module provides configuration profiles, auto-selection utilities, and
performance optimization helpers for VLLM-based models in Kura.
"""

import logging
from typing import Dict, Any, Tuple, Optional, Union
import torch

logger = logging.getLogger(__name__)


class VLLMConfigManager:
    """Manages VLLM configuration profiles and auto-selection."""
    
    def __init__(self):
        """Initialize configuration manager."""
        self.gpu_memory_gb = self._detect_gpu_memory()
        self.gpu_count = self._detect_gpu_count()
        
    def _detect_gpu_memory(self) -> float:
        """Detect total GPU memory across all devices."""
        if not torch.cuda.is_available():
            return 0.0
            
        total_memory = 0.0
        for i in range(torch.cuda.device_count()):
            try:
                props = torch.cuda.get_device_properties(i)
                total_memory += props.total_memory / (1024**3)  # Convert to GB
            except Exception as e:
                logger.warning(f"Failed to get GPU {i} properties: {e}")
                
        return total_memory
        
    def _detect_gpu_count(self) -> int:
        """Detect number of available GPUs."""
        if torch.cuda.is_available():
            return torch.cuda.device_count()
        return 0
        
    def get_optimal_config(
        self,
        n_conversations: int,
        task_type: str = "summarization",
        prefer_quality: bool = False,
        prefer_speed: bool = False
    ) -> Dict[str, Any]:
        """Get optimal configuration for given requirements.
        
        Args:
            n_conversations: Number of conversations to process
            task_type: Type of task ("summarization" or "embedding")
            prefer_quality: Prioritize quality over speed
            prefer_speed: Prioritize speed over quality
            
        Returns:
            Configuration dictionary for the chosen model
        """
        if task_type == "embedding":
            return self._get_embedding_config(n_conversations, prefer_quality, prefer_speed)
        elif task_type == "summarization":
            return self._get_summarization_config(n_conversations, prefer_quality, prefer_speed)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
            
    def _get_embedding_config(
        self, 
        n_conversations: int, 
        prefer_quality: bool, 
        prefer_speed: bool
    ) -> Dict[str, Any]:
        """Get embedding model configuration."""
        # Base configurations
        configs = {
            "fast": {
                "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                "batch_size": 256,
                "max_length": 256,
                "use_sentence_transformers": True,
            },
            "balanced": {
                "model_name": "BAAI/bge-base-en-v1.5",
                "batch_size": 128,
                "max_length": 512,
                "use_sentence_transformers": True,
            },
            "quality": {
                "model_name": "BAAI/bge-large-en-v1.5",
                "batch_size": 64,
                "max_length": 512,
                "use_sentence_transformers": True,
            }
        }
        
        # Choose configuration based on preferences and scale
        if prefer_speed or (n_conversations > 100000 and not prefer_quality):
            config = configs["fast"]
        elif prefer_quality or n_conversations < 10000:
            config = configs["quality"]
        else:
            config = configs["balanced"]
            
        # Optimize batch size based on available GPU memory
        if self.gpu_memory_gb > 0:
            # Scale batch size with available memory
            memory_factor = min(2.0, self.gpu_memory_gb / 8.0)  # Baseline: 8GB GPU
            config["batch_size"] = int(config["batch_size"] * memory_factor)
            config["device"] = "cuda"
        else:
            # CPU configuration
            config["batch_size"] = min(16, config["batch_size"])
            config["device"] = "cpu"
            
        logger.info(f"Selected embedding config: {config['model_name']} with batch_size={config['batch_size']}")
        return config
        
    def _get_summarization_config(
        self,
        n_conversations: int,
        prefer_quality: bool,
        prefer_speed: bool
    ) -> Dict[str, Any]:
        """Get summarization model configuration."""
        # Base configurations
        configs = {
            "small": {
                "model_name": "microsoft/phi-2",
                "tensor_parallel_size": 1,
                "max_num_seqs": 64,
                "gpu_memory_utilization": 0.8,
                "max_model_len": 2048,
            },
            "medium": {
                "model_name": "meta-llama/Llama-2-7b-chat-hf",
                "tensor_parallel_size": 1,
                "max_num_seqs": 128,
                "gpu_memory_utilization": 0.9,
                "max_model_len": 4096,
            },
            "large": {
                "model_name": "meta-llama/Llama-2-13b-chat-hf",
                "tensor_parallel_size": min(2, self.gpu_count),
                "max_num_seqs": 256,
                "gpu_memory_utilization": 0.9,
                "max_model_len": 4096,
            },
            "xl": {
                "model_name": "meta-llama/Llama-2-70b-chat-hf",
                "tensor_parallel_size": min(4, max(2, self.gpu_count)),
                "max_num_seqs": 512,
                "gpu_memory_utilization": 0.9,
                "max_model_len": 4096,
            }
        }
        
        # Memory requirements (approximate, in GB)
        memory_requirements = {
            "small": 4,
            "medium": 16,
            "large": 32,
            "xl": 160
        }
        
        # Choose configuration based on preferences, scale, and available resources
        if prefer_speed or self.gpu_memory_gb < 16:
            chosen_config = "small"
        elif prefer_quality and self.gpu_memory_gb >= 160:
            chosen_config = "xl"
        elif prefer_quality and self.gpu_memory_gb >= 32:
            chosen_config = "large"
        elif n_conversations > 100000 and self.gpu_memory_gb >= 32:
            chosen_config = "large"
        elif n_conversations > 10000 and self.gpu_memory_gb >= 16:
            chosen_config = "medium"
        else:
            chosen_config = "small"
            
        # Verify memory requirements
        required_memory = memory_requirements[chosen_config]
        if self.gpu_memory_gb < required_memory:
            # Fallback to smaller model
            fallbacks = ["small", "medium", "large", "xl"]
            for fallback in fallbacks:
                if memory_requirements[fallback] <= self.gpu_memory_gb:
                    chosen_config = fallback
                    logger.warning(
                        f"Insufficient GPU memory ({self.gpu_memory_gb:.1f}GB) for preferred config. "
                        f"Falling back to {chosen_config}"
                    )
                    break
            else:
                # No suitable GPU configuration found
                raise RuntimeError(
                    f"Insufficient GPU memory ({self.gpu_memory_gb:.1f}GB) for any VLLM configuration. "
                    f"Minimum required: {memory_requirements['small']}GB"
                )
                
        config = configs[chosen_config].copy()
        
        # Adjust tensor parallelism based on available GPUs
        if config["tensor_parallel_size"] > self.gpu_count:
            config["tensor_parallel_size"] = max(1, self.gpu_count)
            logger.warning(
                f"Reduced tensor_parallel_size to {config['tensor_parallel_size']} "
                f"due to limited GPU count ({self.gpu_count})"
            )
            
        logger.info(
            f"Selected summarization config: {config['model_name']} "
            f"with tensor_parallel_size={config['tensor_parallel_size']}, "
            f"max_num_seqs={config['max_num_seqs']}"
        )
        return config
        
    def estimate_cost_and_time(
        self,
        n_conversations: int,
        config: Dict[str, Any],
        task_type: str = "summarization"
    ) -> Tuple[float, float]:
        """Estimate processing cost and time.
        
        Args:
            n_conversations: Number of conversations
            config: Model configuration
            task_type: Type of task
            
        Returns:
            Tuple of (estimated_hours, estimated_cost_usd)
        """
        if task_type == "embedding":
            return self._estimate_embedding_cost_time(n_conversations, config)
        else:
            return self._estimate_summarization_cost_time(n_conversations, config)
            
    def _estimate_embedding_cost_time(
        self, 
        n_conversations: int, 
        config: Dict[str, Any]
    ) -> Tuple[float, float]:
        """Estimate embedding processing cost and time."""
        # Rough estimates based on model size and batch size
        batch_size = config.get("batch_size", 64)
        
        if "large" in config.get("model_name", "").lower():
            seconds_per_batch = 2.0  # Large model
        elif "base" in config.get("model_name", "").lower():
            seconds_per_batch = 1.0  # Base model
        else:
            seconds_per_batch = 0.5  # Small model
            
        # Adjust for device
        if config.get("device") == "cpu":
            seconds_per_batch *= 5  # CPU is much slower
            
        total_batches = (n_conversations + batch_size - 1) // batch_size
        estimated_hours = (total_batches * seconds_per_batch) / 3600
        
        # Cost estimation (GPU rental)
        if config.get("device") == "cuda":
            cost_per_hour = 3.0  # A100 40GB rental cost
            estimated_cost = estimated_hours * cost_per_hour
        else:
            estimated_cost = 0.0  # Local CPU
            
        return estimated_hours, estimated_cost
        
    def _estimate_summarization_cost_time(
        self,
        n_conversations: int,
        config: Dict[str, Any]
    ) -> Tuple[float, float]:
        """Estimate summarization processing cost and time."""
        max_num_seqs = config.get("max_num_seqs", 64)
        
        # Rough estimates based on model size
        model_name = config.get("model_name", "").lower()
        if "70b" in model_name:
            seconds_per_conversation = 15.0
            cost_per_hour = 24.0  # 4x A100 80GB
        elif "13b" in model_name:
            seconds_per_conversation = 5.0
            cost_per_hour = 12.0  # 2x A100 40GB
        elif "7b" in model_name:
            seconds_per_conversation = 2.0
            cost_per_hour = 3.0   # 1x A100 40GB
        else:  # phi-2 or smaller
            seconds_per_conversation = 1.0
            cost_per_hour = 1.5   # 1x RTX 4090
            
        # Account for batching efficiency
        batching_efficiency = min(max_num_seqs / 64, 2.0)  # Up to 2x speedup
        seconds_per_conversation /= batching_efficiency
        
        estimated_hours = (n_conversations * seconds_per_conversation) / 3600
        estimated_cost = estimated_hours * cost_per_hour
        
        return estimated_hours, estimated_cost
        
    def print_recommendation(
        self,
        n_conversations: int,
        task_type: str = "summarization",
        **preferences
    ):
        """Print configuration recommendation with cost/time estimates."""
        config = self.get_optimal_config(n_conversations, task_type, **preferences)
        hours, cost = self.estimate_cost_and_time(n_conversations, config, task_type)
        
        print(f"\n🔧 VLLM Configuration Recommendation for {task_type}")
        print(f"📊 Scale: {n_conversations:,} conversations")
        print(f"💾 Available GPU Memory: {self.gpu_memory_gb:.1f}GB across {self.gpu_count} GPUs")
        print(f"\n🎯 Recommended Configuration:")
        
        if task_type == "embedding":
            print(f"   Model: {config['model_name']}")
            print(f"   Batch Size: {config['batch_size']}")
            print(f"   Device: {config['device']}")
            print(f"   Max Length: {config['max_length']}")
        else:
            print(f"   Model: {config['model_name']}")
            print(f"   Tensor Parallel Size: {config['tensor_parallel_size']}")
            print(f"   Max Concurrent Sequences: {config['max_num_seqs']}")
            print(f"   GPU Memory Utilization: {config['gpu_memory_utilization']:.1f}")
            
        print(f"\n⏱️  Estimated Processing Time: {hours:.1f} hours")
        print(f"💰 Estimated Cost (GPU rental): ${cost:.2f}")
        
        if cost > 50:
            print(f"💡 Consider using a smaller model or API-based approach for cost savings")
        elif hours > 24:
            print(f"💡 Consider using a larger model or more GPUs for faster processing")
            
        print(f"\n📋 Usage Example:")
        if task_type == "embedding":
            print(f"   from kura.vllm_embedding import VLLMEmbeddingModel")
            print(f"   model = VLLMEmbeddingModel(**{config})")
        else:
            print(f"   from kura.vllm_summarisation import VLLMSummaryModel")
            print(f"   model = VLLMSummaryModel(**{config})")


# Helper functions for easy access
def auto_select_models(
    n_conversations: int,
    prefer_quality: bool = False,
    prefer_speed: bool = False
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Auto-select optimal embedding and summarization models.
    
    Args:
        n_conversations: Number of conversations to process
        prefer_quality: Prioritize quality over speed/cost
        prefer_speed: Prioritize speed over quality
        
    Returns:
        Tuple of (embedding_config, summarization_config)
    """
    manager = VLLMConfigManager()
    
    embedding_config = manager.get_optimal_config(
        n_conversations, "embedding", prefer_quality, prefer_speed
    )
    summarization_config = manager.get_optimal_config(
        n_conversations, "summarization", prefer_quality, prefer_speed
    )
    
    return embedding_config, summarization_config


def should_use_vllm(
    n_conversations: int,
    api_cost_per_1k: float = 1.0
) -> bool:
    """Determine if VLLM would be more cost-effective than APIs.
    
    Args:
        n_conversations: Number of conversations
        api_cost_per_1k: API cost per 1000 conversations
        
    Returns:
        True if VLLM is recommended
    """
    # Estimate API cost
    api_cost = (n_conversations / 1000) * api_cost_per_1k
    
    # Estimate VLLM cost
    manager = VLLMConfigManager()
    try:
        config = manager.get_optimal_config(n_conversations, "summarization")
        _, vllm_cost = manager.estimate_cost_and_time(n_conversations, config)
        
        # Include setup/maintenance overhead
        vllm_cost += 50  # Setup cost
        
        return vllm_cost < api_cost * 0.8  # 20% buffer for complexity
    except Exception:
        # If VLLM setup isn't feasible, use APIs
        return False