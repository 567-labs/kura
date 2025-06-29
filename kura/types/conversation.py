from pydantic import BaseModel
from datetime import datetime
from typing import Literal, Union, Callable
import json
import importlib
from tqdm import tqdm

metadata_dict = dict[
    str, Union[str, int, float, bool, list[str], list[int], list[float]]
]


class Message(BaseModel):
    created_at: datetime
    role: Literal["user", "assistant"]
    content: str


class Conversation(BaseModel):
    chat_id: str
    created_at: datetime
    messages: list[Message]
    metadata: metadata_dict

    @classmethod
    def generate_conversation_dump(
        cls, conversations: list["Conversation"], file_path: str
    ) -> None:
        with open(file_path, "w") as f:
            json.dump(
                [
                    conversation.model_dump(mode="json")
                    for conversation in conversations
                ],
                f,
            )

    @classmethod
    def from_conversation_dump(cls, file_path: str) -> list["Conversation"]:
        with open(file_path, "r") as f:
            return [
                Conversation(**conversation)  # ty: ignore
                for conversation in json.load(f)
            ]

    @classmethod
    def from_hf_dataset(
        cls,
        dataset_name: str,
        split: str = "train",
        max_conversations: Union[int, None] = None,
        chat_id_fn=lambda x: x["chat_id"],
        created_at_fn=lambda x: x["created_at"],
        messages_fn=lambda x: x["messages"],
        metadata_fn=lambda x: {},
    ) -> list["Conversation"]:
        if importlib.util.find_spec("datasets") is None:  # type: ignore
            raise ImportError(
                "Please install hf datasets to load conversations from a dataset"
            )
        from datasets import load_dataset  # type: ignore

        if max_conversations:
            dataset = load_dataset(dataset_name, split=split, streaming=True).take(  # type: ignore
                max_conversations
            )
        else:
            dataset = load_dataset(dataset_name, split=split, streaming=True)

        return [
            Conversation(
                chat_id=chat_id_fn(item),
                created_at=created_at_fn(item),
                messages=messages_fn(item),
                metadata=metadata_fn(item),
            )
            for item in tqdm(dataset, desc="Loading Conversations")
        ]

    @classmethod
    def from_claude_conversation_dump(
        cls,
        file_path: str,
        metadata_fn: Callable[[dict], metadata_dict] = lambda x: {},
    ) -> list["Conversation"]:
        with open(file_path, "r") as f:
            return [
                Conversation(
                    chat_id=conversation["uuid"],
                    created_at=conversation["created_at"],
                    messages=[
                        Message(
                            created_at=datetime.fromisoformat(
                                message["created_at"].replace("Z", "+00:00")
                            ),
                            role="user"
                            if message["sender"] == "human"
                            else "assistant",
                            content="\n".join(
                                [
                                    item["text"]
                                    for item in message["content"]
                                    if item["type"] == "text"
                                ]
                            ),
                        )
                        for message in sorted(
                            conversation["chat_messages"],
                            key=lambda x: (
                                datetime.fromisoformat(
                                    x["created_at"].replace("Z", "+00:00")
                                ),
                                0 if x["sender"] == "human" else 1,
                            ),
                        )
                    ],
                    metadata=metadata_fn(conversation),
                )
                for conversation in json.load(f)
            ]
