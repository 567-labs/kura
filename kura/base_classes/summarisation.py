from abc import ABC, abstractmethod
from typing import Type, TypeVar

from kura.types import Conversation, GeneratedSummary, ConversationSummary


T = TypeVar("T", bound=GeneratedSummary)


class BaseSummaryModel(ABC):
    """
    Minimal base class for summary models following principles.

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
        conversations: list[Conversation],
        prompt: str,
        *,
        # All configuration exposed as parameters (not buried in class)
        response_schema: Type[T] = GeneratedSummary,
        temperature: float = 0.2,
        **kwargs,
    ) -> list[ConversationSummary]:
        """
        Summarise conversations with configurable parameters.

        This method implements pure summarization logic, converting conversations
        to structured summaries using the specified response schema.

        Args:
            conversations: List of conversations to summarize
            response_schema: Pydantic model class for structured LLM output
            prompt_template: Custom prompt template (None = use model default)
            temperature: LLM temperature for generation
            **kwargs: Additional model-specific parameters (max_tokens, etc.)

        Returns:
            List of raw model outputs using the specified response_schema

        Example:
            >>> model = SummaryModel()
            >>> summaries = await model.summarise(
            ...     conversations=my_conversations,
            ...     response_schema=DetailedSummary,  # Custom schema
            ...     temperature=0.1
            ... )
        """
        pass

    @property
    @abstractmethod
    def checkpoint_filename(self) -> str:
        """Return the filename to use for checkpointing this model's output."""
        pass
