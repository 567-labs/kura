from typing import Optional, Type
import asyncio
import logging

import instructor
from tqdm.asyncio import tqdm_asyncio
from pydantic import BaseModel
from rich.console import Console

from kura.base_classes import BaseSummaryModel
from kura.types import Conversation, ConversationSummary
from kura.types.summarisation import GeneratedSummary

logger = logging.getLogger(__name__)


class SummaryModel(BaseSummaryModel):
    """
    Instructor-based summary model following

    This implementation follows the embedding.py pattern:
    - Core configuration in constructor (model, concurrency, API keys)
    - Per-use configuration as method parameters (schema, prompts, temperature)
    - Clean separation of concerns (no extractors - handled separately)
    """

    def __init__(
        self,
        model: str = "openai/gpt-4o-mini",
        max_concurrent_requests: int = 50,
        checkpoint_filename: str = "summaries.jsonl",
        console: Optional[Console] = None,
    ):
        """
        Initialize SummaryModel with core configuration.

        Per-use configuration (schemas, prompts, temperature) are method parameters.

        Args:
            model: model identifier (e.g., "openai/gpt-4o-mini")
            max_concurrent_requests: Maximum concurrent API requests
            **kwargs: Additional model-specific parameters (API keys, etc.)
        """
        self.model = model
        self.max_concurrent_requests = max_concurrent_requests
        self.checkpoint_filename = checkpoint_filename
        self.console = console

        logger.info(
            f"Initialized SummaryModel with model={model}, max_concurrent_requests={max_concurrent_requests}"
        )

    def checkpoint_filename(self) -> str:
        return self.checkpoint_filename

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

        if not self.console:
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
        else:
            # Rich console progress tracking with live summary display
            summaries = await self._summarise_with_console(
                conversations,
                client=client,
                response_schema=response_schema,
                prompt_template=prompt_template,
                temperature=temperature,
                **kwargs,
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
        client,
        response_schema: Type[BaseModel],
        prompt_template: str,
        temperature: float,
        **kwargs,
    ) -> ConversationSummary:
        """
        Private method to summarise a single conversation.
        """
        logger.debug(
            f"Starting summarization of conversation {conversation.chat_id} with {len(conversation.messages)} messages"
        )

        async with self.semaphore:  # type: ignore
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

    async def _summarise_with_console(
        self,
        conversations: list[Conversation],
        *,
        client,
        response_schema: Type[BaseModel],
        prompt_template: str,
        temperature: float,
        **kwargs,
    ) -> list[ConversationSummary]:
        """
        Summarise conversations with full-screen Rich console display showing progress and latest 3 results.
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
                    preview_text.append(f"chat_id: {summary.chat_id}\n", style="cyan")
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
                    prompt_template=prompt_template,
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
