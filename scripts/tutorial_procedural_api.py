import time
import asyncio
from contextlib import contextmanager

from kura.base_classes import BaseCheckpointManager
from kura import (
    summarise_conversations,
    generate_base_clusters_from_conversation_summaries,
    reduce_clusters_from_base_clusters,
    reduce_dimensionality_from_clusters,
)

from kura.checkpoints import (
    HFDatasetCheckpointManager,
    JSONLCheckpointManager,
    ParquetCheckpointManager,
)
from kura.types import Conversation
from kura.summarisation import SummaryModel
from kura.k_means import MiniBatchKmeansClusteringMethod
from kura.cluster import ClusterDescriptionModel
from kura.meta_cluster import MetaClusterModel
from kura.dimensionality import HDBUMAP
from rich.console import Console


class TimerManager:
    """A timer class that collects timing data for review."""

    def __init__(self):
        self.timings = {}

    @contextmanager
    def timer(self, message):
        """Context manager to time operations and store results."""
        start_time = time.time()
        yield
        end_time = time.time()
        duration = end_time - start_time
        self.timings[message] = duration
        print(f"{message} took {duration:.2f} seconds")

    def print_summary(self):
        """Print a summary of all collected timings."""
        print(f"\n{'=' * 60}")
        print(f"{'TIMING SUMMARY':^60}")
        print(f"{'=' * 60}")

        total_time = sum(self.timings.values())

        for operation, duration in self.timings.items():
            percentage = (duration / total_time * 100) if total_time > 0 else 0
            print(f"{operation:<40} {duration:>8.2f}s ({percentage:>5.1f}%)")

        print(f"{'-' * 60}")
        print(f"{'Total Time':<40} {total_time:>8.2f}s")
        print(f"{'=' * 60}\n")


# Create a global timer manager instance
timer_manager = TimerManager()


def show_section_header(title):
    """Display a formatted section header."""
    print(f"\n{'=' * 60}")
    print(f"{title:^60}")
    print(f"{'=' * 60}\n")


console = Console()
summary_model = SummaryModel(console=console, max_concurrent_requests=100)
CHECKPOINT_DIR = "./tutorial_checkpoints_2"

minibatch_kmeans_clustering = MiniBatchKmeansClusteringMethod(
    clusters_per_group=10,  # Target items per cluster
    batch_size=1000,  # Mini-batch size for processing
    max_iter=100,  # Maximum iterations
    random_state=42,  # Random seed for reproducibility
)

cluster_model = ClusterDescriptionModel(
    console=console,
)
meta_cluster_model = MetaClusterModel(console=console, max_concurrent_requests=100)
dimensionality_model = HDBUMAP()

with timer_manager.timer("Loading sample conversations"):
    conversations = Conversation.from_hf_dataset(
        "ivanleomk/synthetic-gemini-conversations", split="train"
    )

print(f"Loaded {len(conversations)} conversations successfully!\n")

with timer_manager.timer("Saving conversations to JSON"):
    import json
    import os

    # Ensure checkpoint directory exists
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Convert conversations to JSON format
    conversations_data = [conv.model_dump() for conv in conversations]

    # Save to conversations.json
    with open(f"{CHECKPOINT_DIR}/conversations.json", "w") as f:
        json.dump(conversations_data, f, indent=2, default=str)

print(
    f"Saved {len(conversations)} conversations to {CHECKPOINT_DIR}/conversations.json\n"
)

checkpoint_manager = HFDatasetCheckpointManager(CHECKPOINT_DIR, enabled=True)


async def process(checkpoint_manager: BaseCheckpointManager):
    slug = f"{checkpoint_manager.__class__.__name__}"
    """Process conversations using checkpoints."""
    print("Step 1: Generating summaries with checkpoints...")
    with timer_manager.timer(f"{slug} - Summarization"):
        summaries = await summarise_conversations(
            conversations,
            model=summary_model,
            checkpoint_manager=checkpoint_manager,
        )
    print(f"Generated {len(summaries)} summaries using checkpoints")

    print("Step 2: Generating clusters with checkpoints...")
    with timer_manager.timer(f"{slug} - Clustering"):
        clusters = await generate_base_clusters_from_conversation_summaries(
            summaries,
            model=cluster_model,
            clustering_method=minibatch_kmeans_clustering,
            checkpoint_manager=checkpoint_manager,
        )
    print(f"Generated {len(clusters)} clusters using checkpoints")

    print("Step 3: Meta clustering with checkpoints...")
    with timer_manager.timer(f"{slug} - Meta clustering"):
        reduced_clusters = await reduce_clusters_from_base_clusters(
            clusters,
            model=meta_cluster_model,
            checkpoint_manager=checkpoint_manager,
        )
    print(f"Reduced to {len(reduced_clusters)} meta clusters using checkpoints")

    print("Step 4: Dimensionality reduction with checkpoints...")
    with timer_manager.timer(f"{slug} - Dimensionality reduction"):
        projected_clusters = await reduce_dimensionality_from_clusters(
            reduced_clusters,
            model=dimensionality_model,
            checkpoint_manager=checkpoint_manager,
        )
    print(f"Generated {len(projected_clusters)} projected clusters using {slug}")

    return reduced_clusters, projected_clusters


for checkpoint_manager in [
    HFDatasetCheckpointManager(f"{CHECKPOINT_DIR}/hf", enabled=True),
    ParquetCheckpointManager(f"{CHECKPOINT_DIR}/parquet", enabled=True),
    JSONLCheckpointManager(f"{CHECKPOINT_DIR}/jsonl", enabled=True),
]:
    print(f"Running with {type(checkpoint_manager).__name__}")
    asyncio.run(process(checkpoint_manager))

timer_manager.print_summary()
