from typing import Optional, Type
import asyncio
import logging

import instructor
from tqdm.asyncio import tqdm_asyncio
from pydantic import BaseModel

from kura.base_classes import BaseSummaryModel
from kura.types import Conversation, ConversationSummary
from kura.types.summarisation import GeneratedSummary

logger = logging.getLogger(__name__)


class SummaryModel(BaseSummaryModel):
    """
    Instructor-based summary model following AGENT.md principles.

    This implementation follows the embedding.py pattern:
    - Core configuration in constructor (model, concurrency, API keys)
    - Per-use configuration as method parameters (schema, prompts, temperature)
    - Clean separation of concerns (no extractors - handled separately)
    """

    def checkpoint_filename(self) -> str:
        """Return the filename to use for checkpointing this model's output."""
        return "summaries.jsonl"

    def __init__(
        self,
        model: str = "openai/gpt-4o-mini",
        max_concurrent_requests: int = 50,
    ):
        """
        Initialize OpenAI summary model with core configuration.

        Following AGENT.md principles, only core model configuration goes here.
        Per-use configuration (schemas, prompts, temperature) are method parameters.

        Args:
            model: OpenAI model identifier (e.g., "openai/gpt-4o-mini")
            concurrent_jobs: Maximum concurrent API requests
            **kwargs: Additional model-specific parameters (API keys, etc.)
        """
        self.model = model
        self.max_concurrent_requests = max_concurrent_requests
        self.client = instructor.from_provider(self.model, async_client=True)

        logger.info(
            f"Initialized OpenAISummaryModel with model={model}, max_concurrent_requests={max_concurrent_requests}"
        )

    async def summarise(
        self,
        conversations: list[Conversation],
        *,
        response_schema: Type[BaseModel] = GeneratedSummary,
        prompt_template: Optional[str] = None,
        temperature: float = 0.2,
        **kwargs,
    ) -> list[ConversationSummary]:
        """
        Summarise conversations with configurable parameters.

        Following AGENT.md principles, all important configuration is exposed
        as method parameters rather than buried in the implementation.

        Args:
            conversations: List of conversations to summarize
            response_schema: Pydantic model class for structured LLM output
            prompt_template: prompt_template which will be used to summarise the conversations
            temperature: LLM temperature for generation
            **kwargs: Additional model-specific parameters (max_tokens, etc.)

        Returns:
            List of conversation summaries (without extracted properties)
        """
        # Initialize semaphore per-run to match event loop
        self.semaphore = asyncio.Semaphore(self.max_concurrent_requests)

        logger.info(
            f"Starting summarization of {len(conversations)} conversations using model {self.model}"
        )

        if prompt_template is None:
            prompt_template = self._get_default_prompt_template()

        client = instructor.from_provider(self.model, async_client=True)

        # Simple progress tracking with tqdm
        summaries = await tqdm_asyncio.gather(
            *[
                self._summarise_single_conversation(
                    conversation,
                    client=client,
                    response_schema=response_schema,
                    prompt_template=prompt_template,
                    temperature=temperature,
                    **kwargs,
                )
                for conversation in conversations
            ],
            desc=f"Summarising {len(conversations)} conversations",
        )

        logger.info(
            f"Completed summarization of {len(conversations)} conversations, produced {len(summaries)} summaries"
        )
        return summaries

    def _get_default_prompt_template(self) -> str:
        """
        Get the default prompt template for conversation summarization.

        Based on the Clio paper:
        https://assets.anthropic.com/m/7e1ab885d1b24176/original/Clio-Privacy-Preserving-Insights-into-Real-World-AI-Use.pdf
        """
        return """
The following is a conversation between an AI assistant and a user:

<messages>
{% for message in messages %}
<message>{{message.role}}: {{message.content}}</message>
{% endfor %}
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


Remember that
- Summaries should be concise and short. They should each be at most 1-2 sentences and at most 30 words.
- Summaries should start with "The user's overall request for the assistant is to"
- Make sure to omit any personally identifiable information (PII), like names, locations, phone numbers, email addressess, company names and so on.
- Make sure to indicate specific details such as programming languages, frameworks, libraries and so on which are relevant to the task.
        """.strip()

    async def _summarise_single_conversation(
        self,
        conversation: Conversation,
        *,
        client,  # ✅ Shared client passed from parent method
        response_schema: Type[BaseModel],
        prompt_template: str,
        temperature: float,
        **kwargs,
    ) -> ConversationSummary:
        """
        Private method to summarise a single conversation.

        Uses configurable parameters and shared client passed from the public summarise method.
        Following AGENT.md principles, all behavior is controlled by parameters.
        """
        logger.debug(
            f"Starting summarization of conversation {conversation.chat_id} with {len(conversation.messages)} messages"
        )

        async with self._semaphore:  # type: ignore
            try:
                resp = await client.chat.completions.create(  # type: ignore
                    temperature=temperature,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt_template,
                        },
                    ],
                    context={"messages": conversation.messages},
                    response_model=response_schema,
                    **kwargs,
                )
                logger.debug(
                    f"Successfully generated summary for conversation {conversation.chat_id}"
                )
            except Exception as e:
                logger.error(
                    f"Failed to generate summary for conversation {conversation.chat_id}: {e}"
                )
                raise

        # Create summary with basic metadata (no extractors - handled separately)
        summary = ConversationSummary(
            chat_id=conversation.chat_id,
            **resp.model_dump(),
            metadata={
                "conversation_turns": len(conversation.messages),
                **conversation.metadata,
            },
        )

        logger.debug(
            f"Completed summarization of conversation {conversation.chat_id} - concerning_score: {getattr(resp, 'concerning_score', None)}, user_frustration: {getattr(resp, 'user_frustration', None)}"
        )
        return summary
