from kura.types import Conversation, Message
from kura import (
    summarise_conversations,
    reduce_clusters_from_base_clusters,
    generate_base_clusters_from_conversation_summaries,
    ClusterDescriptionModel,
    MetaClusterModel,
    SummaryModel,
    CheckpointManager,
)
from kura.hdbscan import HDBSCANClusteringMethod
from kura.checkpoints import JSONLCheckpointManager
from datasets import load_dataset
from datetime import datetime

# Use fixed datetime for consistent caching
FIXED_TIME = datetime(2024, 1, 1)
from kura.dimensionality import HDBUMAP
import asyncio
import logfire
from rich.console import Console
import time

from kura.v1.kura import reduce_dimensionality_from_clusters

logfire.configure(
    send_to_logfire=True,
    console=False,
    token="pylf_v1_us_FLfD2wbyzDbjBsxPWdzF4Vx3Dm6JRCBXMJMrPFJcYttT",
)
logfire.instrument_openai()

dataset = load_dataset("lmsys/mt_bench_human_judgments", split="human")

conversations = []

for idx, item in enumerate(dataset):
    models = ["model_a", "model_b"]
    for model in models:
        conversations.append(
            Conversation(
                chat_id=f"{item['question_id']}_{item[model]}_{idx}",
                created_at=FIXED_TIME,
                messages=[
                    Message(
                        role=message["role"],
                        content=message["content"],
                        created_at=FIXED_TIME,
                    )
                    for message in item[f"conversation_{model.replace('model_', '')}"]
                ],
                metadata={
                    "model": item[model],
                    "winner": item["winner"] == model,
                    "question_id": item["question_id"],
                    "opponent_model": item[
                        f"model_{'a' if model == 'model_b' else 'b'}"
                    ],
                },
            )
        )


# Load conversations
Conversation.generate_conversation_dump(
    conversations, "./mt_bench_human_judgments.json"
)
conversations = Conversation.from_conversation_dump("./mt_bench_human_judgments.json")[
    :100
]

console = Console()
# Set up models
summary_model = SummaryModel(
    console=console,
    model="openai/gpt-4.1-mini",
    cache_dir="./cache/summaries",
)
cluster_model = ClusterDescriptionModel(
    console=console,
    model="openai/gpt-4.1-mini",
)
meta_cluster_model = MetaClusterModel(
    max_clusters=10,
    console=console,
    cluster_model=HDBSCANClusteringMethod(),
)
dimensionality_model = HDBUMAP()


async def run_timing_test():
    # First run
    print("Starting first run...")
    start_time = time.time()
    with logfire.span("summarise_conversations_run1"):
        summaries1 = await summarise_conversations(
            conversations,
            model=summary_model,
        )
    first_run_time = time.time() - start_time

    # Second run
    print("Starting second run...")
    start_time = time.time()
    with logfire.span("summarise_conversations_run2"):
        summaries2 = await summarise_conversations(
            conversations,
            model=summary_model,
        )
    second_run_time = time.time() - start_time

    # Show timing difference
    print(f"\nTiming Results:")
    print(f"First run:  {first_run_time:.2f} seconds")
    print(f"Second run: {second_run_time:.2f} seconds")
    print(f"Difference: {first_run_time - second_run_time:.2f} seconds")
    print(f"Speedup:    {first_run_time / second_run_time:.2f}x")


if __name__ == "__main__":
    asyncio.run(run_timing_test())
