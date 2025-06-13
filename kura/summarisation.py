from typing import Optional, Type, TypeVar, Union
import asyncio
import logging
import os
import hashlib

import instructor
from instructor.models import KnownModelName
from tqdm.asyncio import tqdm_asyncio
from rich.console import Console
import diskcache


from kura.base_classes import BaseSummaryModel
from kura.checkpoint import CheckpointManager
from kura.types import Conversation, ConversationSummary
from kura.types.summarisation import GeneratedSummary

T = TypeVar("T", bound=GeneratedSummary)

logger = logging.getLogger(__name__)

DEFAULT_SUMMARY_PROMPT = """
The following is a conversation between an AI assistant and a user:

<messages>
{% for message in conversation.messages %}
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
"""


class SummaryModel(BaseSummaryModel):
    """
    Instructor-based summary model for conversation analysis.

    Example - Custom Schema:
        >>> class CustomSummary(GeneratedSummary):
        ...     sentiment: str
        ...     complexity: int
        >>>
        >>> summaries = await model.summarise(
        ...     conversations,
        ...     response_schema=CustomSummary
        ... )
        # sentiment & complexity will be in summaries[0].metadata

    Example - Custom Prompt:
        >>> summaries = await model.summarise(
        ...     conversations,
        ...     prompt="Also assess the technical complexity on a scale of 1-10."
        ... )
    """

    def __init__(
        self,
        model: Union[str, KnownModelName] = "openai/gpt-4o-mini",
        max_concurrent_requests: int = 50,
        checkpoint_filename: str = "summaries",
        console: Optional[Console] = None,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize SummaryModel with core configuration.

        Per-use configuration (schemas, prompts, temperature) are method parameters.

        Args:
            model: model identifier (e.g., "openai/gpt-4o-mini")
            max_concurrent_requests: Maximum concurrent API requests
            cache_dir: Directory for disk cache storage (optional, defaults to no caching)
        """
        self.model = model
        self.max_concurrent_requests = max_concurrent_requests
        self._checkpoint_filename = checkpoint_filename
        self.console = console
        
        # Initialize disk cache only if cache_dir is provided
        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)
            self.cache = diskcache.Cache(cache_dir)
        else:
            self.cache = None

        logger.info(
            f"Initialized SummaryModel with model={model}, max_concurrent_requests={max_concurrent_requests}, cache_dir={cache_dir}"
        )

    @property
    def checkpoint_filename(self) -> str:
        """Return the filename to use for checkpointing this model's output."""
        return self._checkpoint_filename
    
    def _get_cache_key(
        self, 
        conversation: Conversation, 
        response_schema: Type[T], 
        prompt: str, 
        temperature: float,
        **kwargs
    ) -> str:
        """Generate a cache key from conversation messages and parameters."""
        # Create role-content pairs for each message
        message_data = [(msg.role, msg.content) for msg in conversation.messages]
        
        # Include all parameters that affect the output
        cache_components = (
            tuple(message_data),
            response_schema.__name__,
            hashlib.md5(prompt.encode()).hexdigest(),
            temperature,
            self.model
        )
        
        return hashlib.md5(str(cache_components).encode()).hexdigest()

    async def summarise(
        self,
        conversations: list[Conversation],
        prompt: str = DEFAULT_SUMMARY_PROMPT,
        *,
        response_schema: Type[T] = GeneratedSummary,
        temperature: float = 0.2,
        **kwargs,
    ) -> list[ConversationSummary]:
        """
        Summarise conversations with configurable parameters.

        This method uses the CLIO conversation analysis framework, with automatic
        extensibility for custom fields and prompt modifications.

        Args:
            conversations: List of conversations to summarize
            response_schema: Pydantic model class for structured LLM output.
                           Extend GeneratedSummary to add custom fields that will
                           automatically be included in ConversationSummary.metadata
            prompt: Custom prompt for CLIO analysis
            temperature: LLM temperature for generation

        Returns:
            List of ConversationSummary objects with core fields populated and
            any additional fields from extended schemas in metadata

        Example:
            >>> class CustomSummary(GeneratedSummary):
            ...     sentiment: str
            ...     technical_complexity: int
            >>>
            >>> summaries = await model.summarise(
            ...     conversations,
            ...     response_schema=CustomSummary,
            ...     prompt="Rate sentiment and technical complexity 1-10"
            ... )
            >>> # Access core fields
            >>> print(summaries[0].summary)
            >>> # Access custom fields in metadata
            >>> print(summaries[0].metadata["sentiment"])
        """
        # Initialize semaphore per-run to match event loop
        self.semaphore = asyncio.Semaphore(self.max_concurrent_requests)

        logger.info(
            f"Starting summarization of {len(conversations)} conversations using model {self.model}"
        )

        client = instructor.from_provider(self.model, async_client=True)

        if not self.console:
            # Simple progress tracking with tqdm
            summaries = await tqdm_asyncio.gather(
                *[
                    self._summarise_single_conversation(
                        conversation,
                        client=client,
                        response_schema=response_schema,
                        prompt=prompt,
                        temperature=temperature,
                        **kwargs,
                    )
                    for conversation in conversations
                ],
                desc=f"Summarising {len(conversations)} conversations",
            )
        else:
            # Rich console progress tracking with live summary display
            summaries = await self._summarise_with_console(
                conversations,
                client=client,
                response_schema=response_schema,
                prompt=prompt,
                temperature=temperature,
                **kwargs,
            )

        logger.info(
            f"Completed summarization of {len(conversations)} conversations, produced {len(summaries)} summaries"
        )
        return summaries

    async def _summarise_single_conversation(
        self,
        conversation: Conversation,
        *,
        client,
        response_schema: Type[T],
        prompt: str,
        temperature: float,
        **kwargs,
    ) -> ConversationSummary:
        """
        Private method to summarise a single conversation.

        Automatically maps all fields from the response_schema to ConversationSummary:
        - Known GeneratedSummary fields are mapped directly to ConversationSummary fields
        - Additional fields from extended schemas are placed in metadata for extensibility
        """
        logger.debug(
            f"Starting summarization of conversation {conversation.chat_id} with {len(conversation.messages)} messages"
        )

        # Check cache first (if caching is enabled)
        if self.cache is not None:
            cache_key = self._get_cache_key(conversation, response_schema, prompt, temperature, **kwargs)
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Found cached summary for conversation {conversation.chat_id}")
                return cached_result

        async with self.semaphore:  # type: ignore
            try:
                resp = await client.chat.completions.create(  # type: ignore
                    temperature=temperature,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ],
                    context={
                        "conversation": conversation,
                    },
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

        logger.debug(
            f"Completed summarization of conversation {conversation.chat_id} - concerning_score: {getattr(resp, 'concerning_score', None)}, user_frustration: {getattr(resp, 'user_frustration', None)}"
        )

        # Extract response data
        response_dict = resp.model_dump()

        # Known GeneratedSummary fields that map directly to ConversationSummary
        known_fields = {
            "summary",
            "request",
            "topic",
            "languages",
            "task",
            "concerning_score",
            "user_frustration",
            "assistant_errors",
        }

        # Extract known fields for direct mapping
        known_data = {k: v for k, v in response_dict.items() if k in known_fields}

        # Put unknown fields in metadata (for extended GeneratedSummary subclasses)
        extra_fields = {k: v for k, v in response_dict.items() if k not in known_fields}

        result = ConversationSummary(
            chat_id=conversation.chat_id,
            metadata={
                "conversation_turns": len(conversation.messages),
                **conversation.metadata,
                **extra_fields,  # Additional fields from extended schemas
            },
            **known_data,
        )
        
        # Cache the result (if caching is enabled)
        if self.cache is not None:
            self.cache.set(cache_key, result)
            logger.debug(f"Cached summary for conversation {conversation.chat_id}")
        
        return result

    async def _summarise_with_console(
        self,
        conversations: list[Conversation],
        *,
        client,
        response_schema: Type[T],
        prompt: str,
        temperature: float,
        **kwargs,
    ) -> list[ConversationSummary]:
        """
        Summarise conversations with full-screen Rich console display showing progress and latest 3 results.

        Returns ConversationSummary objects with automatic field mapping from response_schema.
        """
        from rich.live import Live
        from rich.layout import Layout
        from rich.panel import Panel
        from rich.text import Text
        from rich.progress import (
            Progress,
            SpinnerColumn,
            TextColumn,
            TaskProgressColumn,
            TimeRemainingColumn,
        )

        summaries = []
        completed_summaries = []
        max_preview_items = 3

        # Create full-screen layout
        layout = Layout()
        layout.split_column(Layout(name="progress", size=3), Layout(name="preview"))

        # Create progress bar
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=self.console,
        )
        task_id = progress.add_task("", total=len(conversations))
        layout["progress"].update(progress)

        def update_preview_display():
            if completed_summaries:
                preview_text = Text()

                for summary in completed_summaries[
                    -max_preview_items:
                ]:  # Show latest 3
                    preview_text.append(
                        f"summary: {summary.summary or 'No summary'}\n", style="white"
                    )
                    concern = summary.concerning_score or 0
                    frustration = summary.user_frustration or 0
                    preview_text.append(
                        f"Concern: {concern}/5, Frustration: {frustration}/5\n\n",
                        style="yellow",
                    )

                layout["preview"].update(
                    Panel(
                        preview_text,
                        title=f"[green]Generated Summaries ({len(completed_summaries)}/{len(conversations)})",
                        border_style="green",
                    )
                )
            else:
                layout["preview"].update(
                    Panel(
                        Text("Waiting for summaries...", style="dim"),
                        title="[yellow]Generated Summaries (0/0)",
                        border_style="yellow",
                    )
                )

        # Initialize preview display
        update_preview_display()

        with Live(layout, console=self.console, refresh_per_second=4):
            # Process conversations concurrently
            tasks = []
            for conversation in conversations:
                coro = self._summarise_single_conversation(
                    conversation,
                    client=client,
                    response_schema=response_schema,
                    prompt=prompt,
                    temperature=temperature,
                    **kwargs,
                )
                tasks.append(coro)

            # Use asyncio.as_completed to show results as they finish
            for i, coro in enumerate(asyncio.as_completed(tasks)):
                try:
                    summary = await coro
                    summaries.append(summary)
                    completed_summaries.append(summary)

                    # Update progress
                    progress.update(task_id, completed=i + 1)

                    # Update preview display
                    update_preview_display()

                except Exception as e:
                    logger.error(f"Failed to summarise conversation: {e}")
                    # Still update progress on error
                    progress.update(task_id, completed=i + 1)
                    update_preview_display()

        return summaries


async def summarise_conversations(
    conversations: list[Conversation],
    *,
    model: BaseSummaryModel,
    response_schema: Type[T] = GeneratedSummary,
    prompt: str = DEFAULT_SUMMARY_PROMPT,
    temperature: float = 0.2,
    checkpoint_manager: Optional[CheckpointManager] = None,
    **kwargs,
) -> list[ConversationSummary]:
    """Generate summaries for a list of conversations using the CLIO framework.

    This is a pure function that takes conversations and a summary model,
    and returns conversation summaries with automatic extensibility.
    Optionally uses checkpointing for efficient re-runs.

    The function works with any model that implements BaseSummaryModel,
    supporting heterogeneous backends (OpenAI, vLLM, Hugging Face, etc.)
    through polymorphism.

    Extensibility Features:
    - **Custom Fields**: Extend GeneratedSummary to add custom analysis fields
    - **Prompt Modification**: Use prompt to modify CLIO analysis
    - **Automatic Mapping**: Extended fields are automatically placed in metadata

    Args:
        conversations: List of conversations to summarize
        model: Model to use for summarization (OpenAI, vLLM, local, etc.)
        response_schema: Pydantic model class for LLM output. Extend GeneratedSummary
                        to add custom fields that will appear in metadata
        prompt: Custom prompt to modify the CLIO analysis
        temperature: LLM temperature for generation
        checkpoint_manager: Optional checkpoint manager for caching

    Returns:
        List of ConversationSummary objects with core CLIO fields and any
        additional fields from extended schemas in metadata

    Example - Basic Usage:
        >>> model = SummaryModel()
        >>> summaries = await summarise_conversations(
        ...     conversations=my_conversations,
        ...     model=model
        ... )

    Example - Custom Analysis:
        >>> class DetailedSummary(GeneratedSummary):
        ...     sentiment: str
        ...     technical_depth: int
        >>>
        >>> summaries = await summarise_conversations(
        ...     conversations=my_conversations,
        ...     model=model,
        ...     response_schema=DetailedSummary,
        ...     prompt="Analyze sentiment and rate technical depth 1-10"
        ... )
        >>> # Custom fields available in metadata
        >>> print(summaries[0].metadata["sentiment"])
    """
    logger.info(
        f"Starting summarization of {len(conversations)} conversations using {type(model).__name__}"
    )

    # Try to load from checkpoint
    if checkpoint_manager:
        cached = checkpoint_manager.load_checkpoint(
            model.checkpoint_filename, ConversationSummary
        )
        if cached:
            logger.info(f"Loaded {len(cached)} summaries from checkpoint")
            return cached

    # Generate raw summaries
    logger.info("Generating new summaries...")
    raw_summaries = await model.summarise(
        conversations,
        response_schema=response_schema,
        temperature=temperature,
        prompt=prompt,
        **kwargs,
    )
    logger.info(f"Generated {len(raw_summaries)} raw summaries")

    # Summaries are already ConversationSummary objects from _summarise_single_conversation
    summaries = raw_summaries
    logger.info(f"Generated {len(summaries)} conversation summaries")

    # Save to checkpoint
    if checkpoint_manager:
        logger.info(f"Saving summaries to checkpoint: {model.checkpoint_filename}")
        checkpoint_manager.save_checkpoint(model.checkpoint_filename, summaries)

    return summaries
