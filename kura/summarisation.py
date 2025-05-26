from asyncio import Semaphore, gather
from typing import Callable, Optional, Union

import instructor
from tqdm.asyncio import tqdm_asyncio
import asyncio

from kura.base_classes import BaseSummaryModel
from kura.types import Conversation, ConversationSummary, ExtractedProperty
from kura.types.summarisation import GeneratedSummary

# Rich imports handled by Kura base class
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from rich.console import Console


class SummaryModel(BaseSummaryModel):
    @property
    def checkpoint_filename(self) -> str:
        """The filename to use for checkpointing this model's output."""
        return "summaries.jsonl"
    
    def __init__(
        self,
        model: str = "openai/gpt-4o-mini",
        max_concurrent_requests: int = 50,
        extractors: list[
            Callable[
                [Conversation, Semaphore],
                Union[ExtractedProperty, list[ExtractedProperty]]
            ]
        ] = [],
        console: Optional['Console'] = None,
        **kwargs, # For future use
    ):
        self.sems = None
        self.extractors = extractors
        self.max_concurrent_requests = max_concurrent_requests
        self.model = model
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
        self.console = console

    async def _gather_with_progress(self, tasks, desc: str = "Processing", disable: bool = False, show_preview: bool = False):
        """Helper method to run async gather with Rich progress bar if available, otherwise tqdm."""
        if self.console and not disable:
            try:
                from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
                from rich.live import Live
                from rich.layout import Layout
                from rich.panel import Panel
                from rich.text import Text
                from rich.errors import LiveError
                
                if show_preview:
                    # Use Live display with progress and preview buffer
                    layout = Layout()
                    layout.split_column(
                        Layout(name="progress", size=3),
                        Layout(name="preview")
                    )
                    
                    preview_buffer = []
                    max_preview_items = 3
                    
                    # Create progress with cleaner display
                    progress = Progress(
                        SpinnerColumn(),
                        TextColumn("[progress.description]{task.description}"),
                        BarColumn(),
                        TaskProgressColumn(),
                        TimeRemainingColumn(),
                        console=self.console
                    )
                    task_id = progress.add_task(f"[cyan]{desc}...", total=len(tasks))
                    layout["progress"].update(progress)
                    
                    try:
                        with Live(layout, console=self.console, refresh_per_second=4):
                            completed_tasks = []
                            for i, task in enumerate(asyncio.as_completed(tasks)):
                                result = await task
                                completed_tasks.append(result)
                                progress.update(task_id, completed=i + 1)
                                
                                # Add to preview buffer if it's a ConversationSummary
                                if hasattr(result, 'summary') and hasattr(result, 'chat_id'):
                                    preview_buffer.append(result)
                                    if len(preview_buffer) > max_preview_items:
                                        preview_buffer.pop(0)
                                    
                                    # Update preview display
                                    preview_text = Text()
                                    for j, summary in enumerate(preview_buffer):
                                        # Color based on user frustration level
                                        frustration_style = {
                                            1: "green",    # Not frustrated
                                            2: "yellow",   # Slightly frustrated
                                            3: "orange3",  # Moderately frustrated
                                            4: "red",      # Very frustrated
                                            5: "red1"      # Extremely frustrated
                                        }.get(summary.user_frustration, "white")

                                        # Color based on concerning score
                                        concern_style = {
                                            1: "green",    # Not concerning
                                            2: "yellow",   # Slightly concerning
                                            3: "orange3",  # Moderately concerning
                                            4: "red",      # Very concerning
                                            5: "red1"      # Extremely concerning
                                        }.get(summary.concerning_score, "white")

                                        preview_text.append(f"Chat {summary.chat_id[:8]}...: ", style="bold blue")
                                        preview_text.append(f"{summary.summary[:100]}...\n", style=frustration_style)
                                        
                                        if summary.request:
                                            preview_text.append(f"Request: {summary.request[:50]}...\n", style=frustration_style)
                                        if summary.languages:
                                            preview_text.append(f"Languages: {', '.join(summary.languages)}\n", style="dim cyan")
                                        if summary.task:
                                            preview_text.append(f"Task: {summary.task[:50]}...\n", style=concern_style)
                                        
                                        # Add frustration and concern indicators
                                        if summary.user_frustration:
                                            preview_text.append(f"Frustration: {'😊' * summary.user_frustration}\n", style=frustration_style)
                                        if summary.concerning_score:
                                            preview_text.append(f"Concern: {'⚠️' * summary.concerning_score}\n", style=concern_style)
                                        
                                        preview_text.append("\n")
                                    
                                    layout["preview"].update(Panel(
                                        preview_text,
                                        title=f"[green]Recent Summaries ({len(preview_buffer)}/{max_preview_items})",
                                        border_style="green"
                                    ))
                            
                            return completed_tasks
                    except LiveError:
                        # If Rich Live fails, fall back to simple progress without Live
                        with progress:
                            completed_tasks = []
                            for i, task in enumerate(asyncio.as_completed(tasks)):
                                result = await task
                                completed_tasks.append(result)
                                progress.update(task_id, completed=i + 1)
                            return completed_tasks
                else:
                    # Regular progress bar without preview
                    progress = Progress(
                        SpinnerColumn(),
                        TextColumn("[progress.description]{task.description}"),
                        BarColumn(),
                        TaskProgressColumn(),
                        TimeRemainingColumn(),
                        console=self.console
                    )
                    
                    with progress:
                        task_id = progress.add_task(f"[cyan]{desc}...", total=len(tasks))
                        
                        completed_tasks = []
                        for i, task in enumerate(asyncio.as_completed(tasks)):
                            result = await task
                            completed_tasks.append(result)
                            progress.update(task_id, completed=i + 1)
                        
                        return completed_tasks
                        
            except (ImportError, LiveError): #type: ignore
                # Rich not available or Live error, fall back to simple print statements
                self.console.print(f"[cyan]Starting {desc}...[/cyan]")
                completed_tasks = []
                for i, task in enumerate(asyncio.as_completed(tasks)):
                    result = await task
                    completed_tasks.append(result)
                    if (i + 1) % max(1, len(tasks) // 10) == 0 or i == len(tasks) - 1:
                        self.console.print(f"[cyan]{desc}: {i + 1}/{len(tasks)} completed[/cyan]")
                self.console.print(f"[green]✓ {desc} completed![/green]")
                return completed_tasks
        else:
            # Use tqdm as fallback when Rich is not available or disabled
            return await tqdm_asyncio.gather(*tasks, desc=desc, disable=disable)

    async def summarise(
        self, conversations: list[Conversation]
    ) -> list[ConversationSummary]:
        # Initialise Semaphore if not already done
        if self.sems is None:
            self.sems = self.semaphore

        summaries = await self._gather_with_progress(
            [
                self.summarise_conversation(conversation)
                for conversation in conversations
            ],
            desc=f"Summarising {len(conversations)} conversations",
            show_preview=True,
        )
        return summaries

    async def apply_hooks(
        self, conversation: Conversation
    ) -> dict[str, Union[str, int, float, bool, list[str], list[int], list[float]]]:
        coros = [
            extractor(conversation, self.semaphore)
            for extractor in self.extractors
        ]
        metadata_extracted = await gather(*coros)  # pyright: ignore

        metadata = {}
        for result in metadata_extracted:
            if isinstance(result, ExtractedProperty):
                if result.name in metadata:
                    raise ValueError(
                        f"Duplicate metadata name: {result.name}. Please use unique names for each metadata property."
                    )

                metadata[result.name] = result.value

            if isinstance(result, list):
                for extracted_property in result:
                    assert isinstance(extracted_property, ExtractedProperty)
                    if extracted_property.name in metadata:
                        raise ValueError(
                            f"Duplicate metadata name: {extracted_property.name}. Please use unique names for each metadata property."
                        )
                    metadata[extracted_property.name] = extracted_property.value

        return metadata

    async def summarise_conversation(
        self, conversation: Conversation
    ) -> ConversationSummary:
        """
        This summarisation model is designed to extract key information from a conversation between an AI assistant and a user.
        It is designed to be used in a pipeline to summarise conversations and extract metadata.

        It is based on the Clio paper:
            https://assets.anthropic.com/m/7e1ab885d1b24176/original/Clio-Privacy-Preserving-Insights-into-Real-World-AI-Use.pdf

        It is designed to be used in a pipeline to summarise conversations and extract metadata.
        """
        client = instructor.from_provider(self.model, async_client=True)
        async with self.semaphore:  # type: ignore
            resp = await client.chat.completions.create(  # type: ignore
                temperature=0.2, # as per the Clio paper
                messages=[
                    {
                        "role": "user",
                        "content": """
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
                        """,
                    },
                ],
                context={"messages": conversation.messages},
                response_model=GeneratedSummary,
            )

        metadata = await self.apply_hooks(conversation)
        return ConversationSummary(
            chat_id=conversation.chat_id,
            summary=resp.summary,
            request=resp.request,
            languages=resp.languages,
            task=resp.task,
            concerning_score=resp.concerning_score,
            user_frustration=resp.user_frustration,
            assistant_errors=resp.assistant_errors,
            metadata={
                "conversation_turns": len(conversation.messages),
                **conversation.metadata,
                **metadata,
            },
        )
