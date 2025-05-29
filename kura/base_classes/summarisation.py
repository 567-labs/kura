from abc import ABC, abstractmethod
from typing import Optional, Type, List

from kura.types import (
    ConversationSummary,
    Conversation,
    GeneratedSummary,
)
from pydantic import BaseModel


class BaseSummaryModel(ABC):
    """
    Minimal base class for summary models following AGENT.md principles.

    This interface focuses on a single responsibility: converting conversations
    to summaries. All configuration (response schemas, prompts, etc.)
    is exposed as method parameters rather than buried in implementation details.

    Following the embedding.py pattern, implementations should:
    - Accept core model configuration in constructor (API keys, model names, concurrency)
    - Accept per-use configuration as method parameters (schemas, prompts, temperature)
    - Provide a checkpoint_filename() method for checkpointing
    """

    @abstractmethod
    async def summarise(
        self,
        conversations: List[Conversation],
        *,
        # ✅ All configuration exposed as parameters (not buried in class)
        response_schema: Type[BaseModel] = GeneratedSummary,
        prompt_template: Optional[str] = None,
        **kwargs,
    ) -> List[ConversationSummary]:
        """
        Summarise conversations with configurable parameters.

        This method implements pure summarization logic, converting conversations
        to structured summaries. Extraction of additional properties should be
        handled by separate procedural functions following AGENT.md principles.

        Args:
            conversations: List of conversations to summarize
            response_schema: Pydantic model class for structured LLM output
            prompt_template: Custom prompt template (None = use model default)
            **kwargs: Additional model-specific parameters (temperature, max_tokens, etc.)

        Returns:
            List of conversation summaries (without extracted properties)

        Example:
            >>> model = OpenAISummaryModel()
            >>> summaries = await model.summarise(
            ...     conversations=my_conversations,
            ...     response_schema=DetailedSummary,  # Custom schema
            ... )
        """
        pass

    @property
    def checkpoint_filename(self) -> str:
        """Return the filename to use for checkpointing this model's output."""
        pass
