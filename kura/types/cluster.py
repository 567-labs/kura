from pydantic import BaseModel, Field, computed_field
import uuid
from typing import Union


class Cluster(BaseModel):
    id: str = Field(
        default_factory=lambda: uuid.uuid4().hex,
    )
    name: str
    description: str
    slug: str = Field(
        ...,
        description="A three-word snake_case summary of what this cluster is about",
    )
    chat_ids: list[str]
    parent_id: Union[str, None]

    @computed_field
    def count(self) -> int:
        return len(self.chat_ids)

    def __str__(self) -> str:
        return f"Name: {self.name}\nDescription: {self.description}"


class GeneratedCluster(BaseModel):
    name: str = Field(
        ...,
        description="A short, imperative sentence (at most ten words) that captures the user's request and distinguishes this cluster from others. Should be specific but reflective of most statements in the group. Examples: 'Brainstorm ideas for a birthday party' or 'Generate blog spam for gambling websites'",
    )
    summary: str = Field(
        ...,
        description="A clear, precise, two-sentence description in past tense that captures the essence of the clustered statements and distinguishes them from contrastive examples. Should be specific to this group without including PII or proper nouns",
    )
    slug: str = Field(
        ...,
        description="A three-word snake_case summary of what this cluster is about. Examples: 'birthday_party_planning', 'gambling_content_generation', 'code_debugging_help'",
    )


class ClusterTreeNode(BaseModel):
    id: str
    name: str
    description: str
    slug: str
    count: int
    children: list[str]
