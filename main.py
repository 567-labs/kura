from kura import (
    CheckpointManager,
    Conversation,
    SummaryModel,
    ClusterModel
    # Procedural Methods
    summarise_conversations,
)
from kura.embedding import OpenAIEmbeddingModel
from rich.console import Console
import asyncio

# Initialise Models
console = Console()
summary_model = SummaryModel(console=console)
cluster_model = ClusterModel(
    embedding_model=OpenAIEmbeddingModel(),
    console=console
)
checkpoint_manager = CheckpointManager(checkpoint_dir="checkpoints")


async def main():
    # Load Conversations
    conversations = Conversation.from_hf_dataset(
        "ivanleomk/synthetic-gemini-conversations", split="train", max_conversations=10
    )
    # Summarise Conversations
    summaries = await summarise_conversations(
        conversations,
        model=summary_model,
        checkpoint_manager=checkpoint_manager,
    )
    # Generate Base Clusters


if __name__ == "__main__":
    asyncio.run(main())
