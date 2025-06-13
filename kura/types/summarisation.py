from pydantic import BaseModel, Field
from typing import Optional, Union


class GeneratedSummary(BaseModel):
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
    languages: Optional[list[str]] = Field(
        None,
        description="Main languages present in the conversation including human and programming languages (e.g., ['english', 'arabic', 'python', 'javascript'])",
    )
    task: Optional[str] = Field(
        None, description="The task the model is being asked to perform"
    )
    concerning_score: Optional[int] = Field(
        None, ge=1, le=5, description="Safety concern rating from 1-5 scale"
    )
    user_frustration: Optional[int] = Field(
        None, ge=1, le=5, description="User frustration rating from 1-5 scale"
    )
    assistant_errors: Optional[list[str]] = Field(
        None, description="List of errors the assistant made"
    )


class ConversationSummary(GeneratedSummary):
    chat_id: str
    metadata: dict
    embedding: Optional[list[float]] = None

    def __repr__(self) -> str:
        result = f"""<summary>{self.summary}</summary>
<topic>{self.topic}</topic>
<request>{self.request}</request>
<task>{self.task}</task>
<languages>{self.languages}</languages>
<assistant_errors>{self.assistant_errors}</assistant_errors>"""
        
        # Add metadata items with type checking
        for key, value in self.metadata.items():
            if isinstance(value, (str, int, float, bool)):
                result += f"\n<{key}>{value}</{key}>"
            elif isinstance(value, list) and all(isinstance(item, (str, int, float)) for item in value):
                result += f"\n<{key}>{value}</{key}>"
        
        return result


class ExtractedProperty(BaseModel):
    name: str
    value: Union[str, int, float, bool, list[str], list[int], list[float]]