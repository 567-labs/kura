from pydantic import BaseModel, Field
from typing import Optional, Union


class GeneratedSummary(BaseModel):
    """Core summary fields that are actually useful for analysis."""
    summary: str = Field(
        ...,
        description="A clear and concise summary of the conversation in at most two sentences, avoiding phrases like 'Based on the conversation' and excluding proper nouns or PII",
    )
    request: Optional[str] = Field(
        None, description="The user's overall request for the assistant"
    )
    topic: Optional[str] = Field(
        None,
        description="The main high-level topic of the conversation (e.g., 'software development', 'creative writing', 'scientific research').",
    )
    concerning_score: Optional[int] = Field(
        None, ge=1, le=5, description="Safety concern rating from 1-5 scale"
    )
    
    # Optional detailed fields (can be excluded for simpler use cases)
    user_frustration: Optional[int] = Field(
        None, ge=1, le=5, description="User frustration rating from 1-5 scale"
    )
    languages: Optional[list[str]] = Field(
        None,
        description="Main languages present in the conversation including human and programming languages (e.g., ['english', 'arabic', 'python', 'javascript'])",
    )
    task: Optional[str] = Field(
        None, description="The task the model is being asked to perform"
    )
    assistant_errors: Optional[list[str]] = Field(
        None, description="List of errors the assistant made"
    )

    def __repr__(self) -> str:
        return f"""<summary>{self.summary}</summary>
<topic>{self.topic}</topic>
<request>{self.request}</request>
<task>{self.task}</task>
<languages>{self.languages}</languages>
<assistant_errors>{self.assistant_errors}</assistant_errors>"""


class ConversationSummary(GeneratedSummary):
    chat_id: str
    metadata: dict
    embedding: Optional[list[float]] = None


class ExtractedProperty(BaseModel):
    name: str
    value: Union[str, int, float, bool, list[str], list[int], list[float]]
