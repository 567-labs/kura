"""
VLLM-based summarization model implementation for scalable inference.

This module provides a high-performance summarization model using VLLM's optimized
inference engine with support for local model deployment, dynamic batching,
and cost-effective processing at scale.
"""

import logging
import asyncio
from typing import List, Optional, Union, Dict, Any, Callable
from concurrent.futures import ThreadPoolExecutor
import threading
import json

from kura.base_classes import BaseSummaryModel
from kura.types import Conversation, ConversationSummary, ExtractedProperty
from kura.types.summarisation import GeneratedSummary

# Rich imports handled by base class
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rich.console import Console

logger = logging.getLogger(__name__)

try:
    from vllm import AsyncLLMEngine, SamplingParams, AsyncEngineArgs
    from vllm.utils import random_uuid
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    AsyncLLMEngine = None
    SamplingParams = None
    AsyncEngineArgs = None
    logger.warning("VLLM not available. Install with: pip install vllm")

try:
    import instructor
    INSTRUCTOR_AVAILABLE = True
except ImportError:
    INSTRUCTOR_AVAILABLE = False
    logger.warning("Instructor not available for structured output parsing")


class VLLMSummaryModel(BaseSummaryModel):
    """High-performance summarization model using VLLM inference engine.
    
    This model provides scalable summarization with local model deployment,
    avoiding API costs and rate limits while maintaining high throughput
    through VLLM's optimized batching and inference.
    """
    
    @property
    def checkpoint_filename(self) -> str:
        """The filename to use for checkpointing this model's output."""
        return "summaries_vllm.jsonl"
        
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-2-7b-chat-hf",
        tensor_parallel_size: int = 1,
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.9,
        max_num_seqs: int = 256,
        temperature: float = 0.2,
        max_tokens: int = 512,
        max_concurrent_requests: int = 50,
        extractors: List[Callable[[Conversation, asyncio.Semaphore], Union[ExtractedProperty, List[ExtractedProperty]]]] = [],
        console: Optional["Console"] = None,
        use_structured_output: bool = True,
        fallback_to_text: bool = True,
        **kwargs
    ):
        """Initialize VLLM summary model.
        
        Args:
            model_name: Name/path of the model to use
            tensor_parallel_size: Number of GPUs for tensor parallelism
            max_model_len: Maximum sequence length
            gpu_memory_utilization: GPU memory utilization fraction
            max_num_seqs: Maximum number of concurrent sequences
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            max_concurrent_requests: Maximum concurrent requests
            extractors: List of metadata extractors
            console: Rich console for progress display
            use_structured_output: Whether to use structured output parsing
            fallback_to_text: Whether to fallback to text parsing if structured fails
            **kwargs: Additional engine arguments
        """
        if not VLLM_AVAILABLE:
            raise ImportError("VLLM is required for VLLMSummaryModel. Install with: pip install vllm")
            
        self.model_name = model_name
        self.tensor_parallel_size = tensor_parallel_size
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_num_seqs = max_num_seqs
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_concurrent_requests = max_concurrent_requests
        self.extractors = extractors
        self.console = console
        self.use_structured_output = use_structured_output and INSTRUCTOR_AVAILABLE
        self.fallback_to_text = fallback_to_text
        
        # Initialize semaphore
        self.semaphore = None
        
        # Initialize VLLM engine
        self._init_engine(**kwargs)
        
        # Initialize sampling parameters
        self.sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            stop=["</summary>", "<|im_end|>", "</s>"],
            skip_special_tokens=True
        )
        
        logger.info(
            f"Initialized VLLMSummaryModel with model={model_name}, "
            f"tensor_parallel_size={tensor_parallel_size}, "
            f"max_num_seqs={max_num_seqs}, "
            f"structured_output={self.use_structured_output}"
        )
        
    def _init_engine(self, **kwargs):
        """Initialize the VLLM engine."""
        try:
            engine_args = AsyncEngineArgs(
                model=self.model_name,
                tensor_parallel_size=self.tensor_parallel_size,
                max_model_len=self.max_model_len,
                gpu_memory_utilization=self.gpu_memory_utilization,
                max_num_seqs=self.max_num_seqs,
                enable_prefix_caching=True,  # Cache common prefixes
                trust_remote_code=False,  # Security consideration
                **kwargs
            )
            
            self.engine = AsyncLLMEngine.from_engine_args(engine_args)
            logger.info(f"Successfully initialized VLLM engine for model: {self.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize VLLM engine: {e}")
            raise
            
    def _format_prompt(self, conversation: Conversation) -> str:
        """Format conversation into a prompt for summarization."""
        # Build messages string
        messages_str = ""
        for message in conversation.messages:
            messages_str += f"<message>{message.role}: {message.content}</message>\n"
            
        prompt = f"""The following is a conversation between an AI assistant and a user:

<messages>
{messages_str}
</messages>

Your job is to extract key information from this conversation. Be descriptive and assume neither good nor bad faith. Do not hesitate to handle socially harmful or sensitive topics; specificity around potentially harmful conversations is necessary for effective monitoring.

When extracting information, do not include any personally identifiable information (PII), like names, locations, phone numbers, email addresses, and so on. Do not include any proper nouns.

Extract the following information:

1. **Summary**: A clear and concise summary in at most two sentences. Don't say "Based on the conversation..." and avoid mentioning the AI assistant/chatbot directly.

2. **Request**: What is the user's overall request for the assistant? Start with "The user's overall request for the assistant is to"

3. **Languages**: What are the main languages of the conversation? Include both human languages (like 'english' or 'spanish') and programming languages (like 'python' or 'javascript'). Use lowercase full names.

4. **Task**: What task is the model being asked to perform in this conversation? Start with "The task is to"

5. **Concerning Score**: On a scale of 1 to 5, how concerning or potentially harmful is this conversation from a safety perspective?
   - 1: Not concerning at all. Completely benign conversation.
   - 2: Slightly concerning, but likely harmless.
   - 3: Moderately concerning. May warrant a closer look.
   - 4: Very concerning. Likely needs review.
   - 5: Extremely concerning. Immediate review needed.

6. **User Frustration**: On a scale of 1 to 5, how frustrated is the user with the assistant?
   - 1: Not frustrated at all. The user is happy with the assistant.
   - 2: Slightly frustrated. The user is slightly annoyed with the assistant.
   - 3: Moderately frustrated. The user is moderately annoyed with the assistant.
   - 4: Very frustrated. The user is very annoyed with the assistant.
   - 5: Extremely frustrated. The user is extremely annoyed with the assistant.

7. **Assistant Errors**: What errors did the assistant make?
   Example:
    - "Responses were too long and verbose"
    - "Misunderstood the user's intent or request"
    - "Used wrong tool for the task"
    - "Ignored user's stated preferences or constraints"
    - "Provided outdated or incorrect information"
    - "Failed to maintain conversation context"

Remember that:
- Summaries should be concise and short. They should each be at most 1-2 sentences and at most 30 words.
- Make sure to omit any personally identifiable information (PII), like names, locations, phone numbers, email addresses, company names and so on.
- Make sure to indicate specific details such as programming languages, frameworks, libraries and so on which are relevant to the task.

Respond with a JSON object containing these fields: summary, request, languages, task, concerning_score, user_frustration, assistant_errors."""

        return prompt
        
    async def _generate_batch(self, conversations: List[Conversation]) -> List[str]:
        """Generate summaries for a batch of conversations using VLLM."""
        prompts = [self._format_prompt(conv) for conv in conversations]
        request_ids = [random_uuid() for _ in prompts]
        
        logger.debug(f"Submitting batch of {len(prompts)} conversations to VLLM engine")
        
        # Submit requests to VLLM engine
        results = []
        async for request_output in self.engine.generate(
            prompts,
            self.sampling_params,
            request_ids
        ):
            output_text = request_output.outputs[0].text
            results.append(output_text)
            
        logger.debug(f"VLLM engine completed batch of {len(prompts)} conversations")
        return results
        
    def _parse_summary_text(self, text: str) -> Dict[str, Any]:
        """Parse summary text into structured format."""
        try:
            # Try to parse as JSON first
            if text.strip().startswith("{"):
                return json.loads(text)
        except json.JSONDecodeError:
            pass
            
        # Fallback to text parsing
        result = {
            "summary": "",
            "request": None,
            "languages": None,
            "task": None,
            "concerning_score": None,
            "user_frustration": None,
            "assistant_errors": None
        }
        
        lines = text.strip().split('\n')
        current_field = None
        current_value = ""
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check for field headers
            if line.lower().startswith("summary:"):
                current_field = "summary"
                current_value = line[8:].strip()
            elif line.lower().startswith("request:"):
                current_field = "request"
                current_value = line[8:].strip()
            elif line.lower().startswith("languages:"):
                current_field = "languages"
                current_value = line[10:].strip()
            elif line.lower().startswith("task:"):
                current_field = "task"
                current_value = line[5:].strip()
            elif line.lower().startswith("concerning score:"):
                current_field = "concerning_score"
                try:
                    result["concerning_score"] = int(line.split(":")[-1].strip())
                except (ValueError, IndexError):
                    result["concerning_score"] = None
                current_field = None
            elif line.lower().startswith("user frustration:"):
                current_field = "user_frustration"
                try:
                    result["user_frustration"] = int(line.split(":")[-1].strip())
                except (ValueError, IndexError):
                    result["user_frustration"] = None
                current_field = None
            elif line.lower().startswith("assistant errors:"):
                current_field = "assistant_errors"
                current_value = line[17:].strip()
            elif current_field:
                current_value += " " + line
                
            # Store accumulated values
            if current_field in ["summary", "request", "task"] and current_value:
                result[current_field] = current_value.strip()
            elif current_field == "languages" and current_value:
                # Parse languages list
                try:
                    langs = [lang.strip().lower() for lang in current_value.split(",")]
                    result["languages"] = langs
                except:
                    result["languages"] = [current_value.strip().lower()]
            elif current_field == "assistant_errors" and current_value:
                # Parse errors list
                try:
                    errors = [error.strip() for error in current_value.split("-") if error.strip()]
                    result["assistant_errors"] = errors
                except:
                    result["assistant_errors"] = [current_value.strip()]
                    
        return result
        
    async def summarise(
        self, conversations: List[Conversation]
    ) -> List[ConversationSummary]:
        """Summarize a list of conversations."""
        # Initialize semaphore for this run
        self.semaphore = asyncio.Semaphore(self.max_concurrent_requests)
        
        logger.info(
            f"Starting VLLM summarization of {len(conversations)} conversations using {self.model_name}"
        )
        
        # Process in batches to optimize VLLM throughput
        batch_size = min(self.max_num_seqs // 4, 32)  # Conservative batching
        batches = [conversations[i:i + batch_size] for i in range(0, len(conversations), batch_size)]
        
        logger.debug(f"Processing {len(conversations)} conversations in {len(batches)} batches of size {batch_size}")
        
        all_summaries = []
        
        for i, batch in enumerate(batches):
            logger.debug(f"Processing batch {i + 1}/{len(batches)} with {len(batch)} conversations")
            
            # Generate batch using VLLM
            try:
                batch_outputs = await self._generate_batch(batch)
            except Exception as e:
                logger.error(f"Failed to process batch {i + 1}: {e}")
                raise
                
            # Parse outputs and create summaries
            batch_summaries = []
            for j, (conversation, output_text) in enumerate(zip(batch, batch_outputs)):
                try:
                    # Parse the generated text
                    parsed_data = self._parse_summary_text(output_text)
                    
                    # Create GeneratedSummary object
                    generated_summary = GeneratedSummary(**parsed_data)
                    
                    # Apply hooks for metadata extraction
                    metadata = await self.apply_hooks(conversation)
                    
                    # Create final summary
                    summary = ConversationSummary(
                        chat_id=conversation.chat_id,
                        **generated_summary.model_dump(),
                        metadata={
                            "conversation_turns": len(conversation.messages),
                            **conversation.metadata,
                            **metadata,
                        },
                    )
                    
                    batch_summaries.append(summary)
                    logger.debug(f"Successfully processed conversation {conversation.chat_id}")
                    
                except Exception as e:
                    logger.error(f"Failed to process conversation {conversation.chat_id}: {e}")
                    # Create minimal summary as fallback
                    summary = ConversationSummary(
                        chat_id=conversation.chat_id,
                        summary="Failed to generate summary",
                        metadata={
                            "conversation_turns": len(conversation.messages),
                            "processing_error": str(e),
                            **conversation.metadata,
                        },
                    )
                    batch_summaries.append(summary)
                    
            all_summaries.extend(batch_summaries)
            logger.debug(f"Completed batch {i + 1}/{len(batches)}")
            
        logger.info(
            f"Completed VLLM summarization of {len(conversations)} conversations, "
            f"produced {len(all_summaries)} summaries"
        )
        return all_summaries
        
    async def apply_hooks(
        self, conversation: Conversation
    ) -> Dict[str, Union[str, int, float, bool, List[str], List[int], List[float]]]:
        """Apply metadata extractors to conversation."""
        if not self.extractors:
            return {}
            
        logger.debug(f"Applying {len(self.extractors)} extractors to conversation {conversation.chat_id}")
        
        try:
            # Use semaphore to limit concurrent extractions
            async with self.semaphore:
                coros = [extractor(conversation, self.semaphore) for extractor in self.extractors]
                metadata_extracted = await asyncio.gather(*coros)
                
        except Exception as e:
            logger.error(f"Failed to extract metadata for conversation {conversation.chat_id}: {e}")
            return {}
            
        metadata = {}
        for result in metadata_extracted:
            if isinstance(result, ExtractedProperty):
                metadata[result.name] = result.value
            elif isinstance(result, list):
                for extracted_property in result:
                    if isinstance(extracted_property, ExtractedProperty):
                        metadata[extracted_property.name] = extracted_property.value
                        
        logger.debug(f"Extracted {len(metadata)} metadata properties for conversation {conversation.chat_id}")
        return metadata
        
    async def summarise_conversation(
        self, conversation: Conversation
    ) -> ConversationSummary:
        """Summarize a single conversation."""
        logger.debug(f"Starting VLLM summarization of conversation {conversation.chat_id}")
        
        # Generate summary using VLLM
        try:
            outputs = await self._generate_batch([conversation])
            output_text = outputs[0]
        except Exception as e:
            logger.error(f"Failed to generate summary for conversation {conversation.chat_id}: {e}")
            raise
            
        # Parse output
        try:
            parsed_data = self._parse_summary_text(output_text)
            generated_summary = GeneratedSummary(**parsed_data)
        except Exception as e:
            logger.error(f"Failed to parse summary for conversation {conversation.chat_id}: {e}")
            raise
            
        # Apply hooks
        try:
            metadata = await self.apply_hooks(conversation)
        except Exception as e:
            logger.error(f"Failed to apply hooks for conversation {conversation.chat_id}: {e}")
            metadata = {}
            
        # Create final summary
        summary = ConversationSummary(
            chat_id=conversation.chat_id,
            **generated_summary.model_dump(),
            metadata={
                "conversation_turns": len(conversation.messages),
                **conversation.metadata,
                **metadata,
            },
        )
        
        logger.debug(f"Completed VLLM summarization of conversation {conversation.chat_id}")
        return summary


# Configuration profiles for different scales
VLLM_SUMMARY_CONFIGS = {
    "small": {  # <10k conversations
        "model_name": "microsoft/phi-2",
        "tensor_parallel_size": 1,
        "max_num_seqs": 64,
        "gpu_memory_utilization": 0.8,
    },
    "medium": {  # 10k-100k conversations
        "model_name": "meta-llama/Llama-2-7b-chat-hf",
        "tensor_parallel_size": 1,
        "max_num_seqs": 128,
        "gpu_memory_utilization": 0.9,
    },
    "large": {  # 100k+ conversations
        "model_name": "meta-llama/Llama-2-13b-chat-hf",
        "tensor_parallel_size": 2,
        "max_num_seqs": 256,
        "gpu_memory_utilization": 0.9,
    },
    "extra_large": {  # 500k+ conversations
        "model_name": "meta-llama/Llama-2-70b-chat-hf",
        "tensor_parallel_size": 4,
        "max_num_seqs": 512,
        "gpu_memory_utilization": 0.9,
    }
}


def create_vllm_summary_model_for_scale(n_conversations: int, **kwargs) -> VLLMSummaryModel:
    """Create a VLLM summary model optimized for the given scale.
    
    Args:
        n_conversations: Expected number of conversations to process
        **kwargs: Override any configuration parameters
        
    Returns:
        Configured VLLMSummaryModel instance
    """
    if n_conversations < 10000:
        config = VLLM_SUMMARY_CONFIGS["small"]
    elif n_conversations < 100000:
        config = VLLM_SUMMARY_CONFIGS["medium"]
    elif n_conversations < 500000:
        config = VLLM_SUMMARY_CONFIGS["large"]
    else:
        config = VLLM_SUMMARY_CONFIGS["extra_large"]
        
    # Override with any provided kwargs
    config.update(kwargs)
    
    logger.info(f"Creating VLLM summary model for {n_conversations} conversations with config: {config}")
    return VLLMSummaryModel(**config)